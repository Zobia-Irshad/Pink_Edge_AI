# Pink Edge AI: SECURITY.md and Repository Security Findings

The policy below is ready to paste, and it uses the repo's real versioning. Zobia-Irshad/Pink_Edge_AI has no published releases, so the only supported version is the current `main` branch, a hackathon-grade prototype. The template's 5.1.x / 4.0.x rows are dropped. Fix three things before or alongside publishing: a committed `pink_edge_cache.db`, a committed `__pycache__/`, and a missing LICENSE and contact address.

## TL;DR
- **Supported versions:** there are no releases or changelog, and the README calls the project "a hackathon-grade demo, not a validated medical device." So the policy supports only the latest `main` commit. The original hackathon code in `Misc/` and any forks or copies are marked unsupported.
- **How to report:** the policy uses GitHub private vulnerability reporting. According to GitHub Docs, "owners and administrators of public repositories" must switch it on under **Settings → Advanced Security → Private vulnerability reporting → Enable**. It forbids public issues and real patient data. It promises acknowledgement within 5 business days, assessment within 14 days and a fix within 90 days.
- **Fix in the repo first:** the SQLite report cache and `__pycache__/` are committed. There is no LICENSE, SECURITY.md, CONTRIBUTING.md or contact email. A near-identical copy, `Faheem078/Pink_Edge_AI-main`, has a committed `roboflow_key.txt`. If that key belongs to your team, revoke it. I could not read file contents (PINs, `.gitignore`, `requirements.txt`) directly, so verify those items locally.

## Key Findings (repository security review — fix before or alongside publishing)

| # | Finding | Status | Severity | Action |
|---|---|---|---|---|
| 1 | `pink_edge_cache.db` (the SQLite report cache) is committed at the repo root. The README says it holds patient reports with pending/synced status and is created on first "Save to Cache", and it never mentions encryption. | Verified (file is in the root listing); contents and encryption not inspected | High if it holds anything other than synthetic data | Inspect it (`sqlite3 pink_edge_cache.db .dump`). Run `git rm --cached pink_edge_cache.db` and add `*.db` to `.gitignore`. If it ever held real patient data, purge it from history (`git filter-repo`) and treat that as a privacy incident. | [github](https://github.com/Zobia-Irshad/Pink_Edge_AI)
| 2 | `roboflow_key.txt` is committed in the near-identical copy `Faheem078/Pink_Edge_AI-main`. The README there says the file is "(gitignored)", which contradicts the repo. It is absent from the Zobia-Irshad root. | Verified that the file is present in that copy; contents not inspected | High if it is a real key | Find out whose key it is. Revoke and regenerate it in Roboflow (Settings → API), because a key that was public must be treated as compromised even after deletion. Ask the copy's owner to remove it. Check this repo's history too: `git log --all -p -- roboflow_key.txt`. | [github](https://github.com/Faheem078/Pink_Edge_AI-main)
| 3 | Default role PINs (LHW `1111`, Senior Radiologist `9999`) are shown on screen, according to prior analysis. A README in the same codebase family says roles are "switchable in the sidebar or via PIN", which may let someone change role without entering a PIN. | Not verified; `auth_manager.py` could not be read | High for any shared or networked deployment | Remove the PINs from the UI. Read PINs from environment variables or secrets and store them as salted hashes (e.g., `hashlib.scrypt`). Add lockout after failed attempts. Make sure the sidebar switch also requires the PIN. | [github](https://github.com/Faheem078/Pink_Edge_AI-main)
| 4 | `__pycache__/` is committed. | Verified | Low (hygiene; may reveal local paths) | Remove it and add `__pycache__/` and `*.pyc` to `.gitignore`. | [github](https://github.com/Zobia-Irshad/Pink_Edge_AI)
| 5 | The repo has no LICENSE, SECURITY.md, CONTRIBUTING.md or CODE_OF_CONDUCT, and no maintainer email in the README or repo "About" section. The README's line that the repo "already has" a .gitignore and license is leftover template text. | Verified at root | Medium (governance) | Publish this SECURITY.md and turn on private vulnerability reporting. Add a LICENSE that is compatible with the model licenses listed in `Documentations/MODEL_SOURCES.md`. | [github](https://github.com/Zobia-Irshad/Pink_Edge_AI)
| 6 | Dependencies appear unpinned. The README says to "pin lighter dependency versions" if deploys struggle. | Inferred; `requirements.txt` could not be read | Medium | Pin exact versions (or use a lock file). Enable Dependabot alerts and updates, which are free for public repos. | [github](https://github.com/Zobia-Irshad/Pink_Edge_AI)
| 7 | `dicom_anonymizer.py` claims to be "HIPAA/GDPR compliant". That is a self-assessment. NEMA's DICOM PS3.15 Annex E itself warns that "conformance to the Basic Application Level Confidentiality Profile does not necessarily guarantee confidentiality." Unsalted hashes of short patient IDs can be reversed by brute force. | Claim verified in README; implementation not inspected | Medium | Reword the claim as "designed to support" HIPAA/GDPR de-identification. Use a keyed hash (HMAC with a secret key) and remove or replace all PS3.15 Annex E attributes, including private tags and text burned into the image pixels. |
| 8 | The README recommends `--server.address 0.0.0.0` and `--server.port 80` for LAN or self-hosting. That serves the app over plain HTTP, with PINs as the only access control. | Verified in README | Medium | Document that the app must sit behind HTTPS and real authentication (or be bound to localhost) before any networked use. | [github](https://github.com/Zobia-Irshad/Pink_Edge_AI)
| 9 | A README in the same codebase patches `st.markdown` globally so that HTML passed with `unsafe_allow_html=True` is auto-dedented. Any user-controlled text (DICOM metadata, override reasons, patient IDs) inserted into those HTML strings without escaping is an XSS risk. | Verified in the `Faheem078` copy's README; not confirmed in this repo's code | Medium | Pass every interpolated value through `html.escape()`. | [github](https://github.com/Faheem078/Pink_Edge_AI-main)
| 10 | `Test Data/` and `Models/*/Data Set/` hold real sample medical images. | Verified (folders present) | Medium (privacy and licensing) | Confirm every image comes from a public, licensed, de-identified dataset and that no DICOM headers carry PHI. Record each source. | [github](https://github.com/Zobia-Irshad/Pink_Edge_AI)
| 11 | The Alibaba Cloud IoT/OSS/ACR sync is simulated. The README says "no real Alibaba credentials are used". | Verified in README | Informational | Keep it out of scope until it is real, and keep credentials out of the repo when it is. | [github](https://github.com/Zobia-Irshad/Pink_Edge_AI)

Housekeeping note: the README's project-layout section points to `App/...`, but the files are at the repo root, and several entries are listed twice. [github](https://github.com/Zobia-Irshad/Pink_Edge_AI) Fix this so reporters and deployers know which paths are real.

## The SECURITY.md file (paste everything inside the block below)

````markdown
# Security Policy

Pink Edge AI is an offline-first clinical-triage **prototype** for mammography, tuberculosis (chest X-ray) and maternal-health (ultrasound) imaging. It ships as a Tkinter desktop app (`GUI.py`) and a Streamlit web app (`streamlit_app.py`) that share one backend (`inference.py`, `offline_cv.py`), a local SQLite report cache, role-based access control (`auth_manager.py`) and a DICOM anonymizer (`dicom_anonymizer.py`). [github](https://github.com/Zobia-Irshad/Pink_Edge_AI)

> **Important:** Pink Edge AI is a hackathon-grade demo. It is **not** a certified or validated medical device, it has not undergone clinical or regulatory review, and it **must not be used to make real clinical decisions or to process real patient data**. Confidence scores and severity mappings are illustrative.

We still take security and privacy seriously, because this code handles medical images and patient-like records. Please follow this policy if you find a problem.

## Supported Versions

This project has no tagged releases. Security fixes are applied only to the latest commit on the `main` branch.

| Version | Supported |
| ------- | :-------: |
| `main` branch (latest commit) | :white_check_mark: |
| Older commits on `main` | :x: — please update to the latest commit |
| Original hackathon submission in `Misc/Pink_Edge_AI-main/` | :x: — reference only |
| Forks, mirrors and copies maintained by others | :x: — report to their owners |

If versioned releases are published in the future, this table will be updated to list which release lines receive fixes.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues, pull requests, discussions or social media.**

### Preferred channel: GitHub private vulnerability reporting

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability** (or open https://github.com/Zobia-Irshad/Pink_Edge_AI/security/advisories/new).
3. Fill in the form. Only you and the maintainers can see the report.

If the button is not available, contact the maintainer privately at **[maintainer email]** or through the GitHub profile [@Zobia-Irshad](https://github.com/Zobia-Irshad), and ask for a private channel. Do not include vulnerability details in a public message.

### What to include

- A short description of the issue and its potential impact (e.g., RBAC bypass, PHI leakage, key exposure).
- The affected component (file, function or UI screen) and the commit hash you tested.
- Edition (desktop or Streamlit), OS, Python version and how you ran it (local, LAN, Streamlit Community Cloud).
- Step-by-step reproduction instructions and a minimal proof of concept.
- Any suggested fix or mitigation.

### Never include real patient data

- **Do not attach, upload or paste real patient data, protected health information (PHI), personally identifiable information (PII) or real DICOM files** in any report.
- Use synthetic images, the images in `Test Data/`, or DICOM files you have generated yourself with fake metadata.
- If your report concerns PHI exposure (for example, the anonymizer leaving a tag in place or the cache leaking records), describe **which field or tag leaked and how**, not the actual values.
- If you accidentally come across what looks like real patient data in this repository, stop, do not copy or share it, and report its location to us privately straight away. We will treat it as a priority incident and remove it, including from git history.

### What to expect

This is a small volunteer project. We aim for the following timelines:

| Stage | Target |
| ----- | ------ |
| Acknowledgement of your report | Within **5 business days** |
| Initial assessment (confirmed / needs info / not a vulnerability) | Within **14 days** |
| Status updates while we work on it | At least every **14 days** |
| Fix for confirmed issues | Within **90 days** (critical issues such as PHI leakage, credential exposure or authentication bypass are prioritised) |
| Public disclosure | After a fix is available, or at 90 days, coordinated with you |

**If the report is accepted,** we will confirm the severity with you, develop a fix privately (if you wish, we can add you as a collaborator on a draft GitHub Security Advisory), merge it to `main`, and publish an advisory describing the issue and the fix. We will credit you unless you prefer to remain anonymous.

**If the report is declined,** for example because it is out of scope, cannot be reproduced or is expected prototype behaviour listed under *Known Limitations*, we will explain why. If you disagree, you are welcome to reply with more information and we will reconsider.

Please give us reasonable time to fix the issue before you disclose it publicly, and coordinate the disclosure date with us.

## Scope

### In scope

- **Authentication and RBAC** (`auth_manager.py`): bypassing the PIN or biometric check, escalating from Lady Health Worker to Senior Radiologist, or overriding AI assessments without the right role.
- **DICOM anonymization** (`dicom_anonymizer.py`): PHI/PII tags not removed, private tags or burned-in text kept, privacy hashes that can be reversed or linked back to a patient, or identifiable data reaching the UI, cache, reports or alerts before anonymization.
- **Local report cache** (`pink_edge_cache.db`): unauthorised read or modification of cached reports, SQL injection, or patient-identifying data stored in plain text.
- **Secrets handling**: Roboflow API keys (`roboflow_key.txt`, the `ROBOFLOW_API_KEY` environment variable, Streamlit `st.secrets`) being logged, displayed, committed or otherwise exposed.
- **GSM/SMS alert module**: the 140-character alert payload (patient ID, location, BI-RADS) exposing more than intended, being forgeable, or being sent to unintended recipients.
- **Streamlit web app** (`streamlit_app.py`): XSS through HTML rendered with `unsafe_allow_html=True`, unsafe file-upload handling, path traversal, or data leaking between sessions.
- **Generated reports** (text/PDF via fpdf2): injection or unintended inclusion of identifying data.
- **Model loading and inference** (`inference.py`): loading untrusted model files or pickles from attacker-controlled paths, or unsafe deserialisation.
- **Dependencies** listed in `requirements.txt` (e.g., Streamlit, PyTorch, Ultralytics, OpenCV, NumPy, Pillow, fpdf2) where a known vulnerability can actually be exploited through Pink Edge AI.
- **Repository contents**: committed secrets, credentials or real patient data anywhere in the repository or its history.

### Out of scope

- The **simulated** Alibaba Cloud IoT / OSS / ACR sync panel and the simulated hardware-diagnostics panel. These use no real credentials or cloud endpoints. [github](https://github.com/Zobia-Irshad/Pink_Edge_AI)
- Vulnerabilities in **third-party services or models themselves**: Roboflow, Hugging Face Hub, the models `sukhmani1303/tuberculosis-vit-model`, `shr3m/fetal-brain-plane-cnn`, `breastcancer-yolov8-78tni` and `hash-maternal-health/1`, Streamlit Community Cloud, or GitHub. Please report these to the relevant vendor.
- **Model accuracy, misclassification or clinical-safety concerns** that are not security issues. Please open a regular issue for these, without patient data.
- Vulnerabilities in dependencies that cannot be reached through this project, or reports from automated scanners with no demonstrated impact.
- Social engineering, phishing, or physical attacks on maintainers, users or devices.
- Denial of service by flooding or resource exhaustion, including the known memory limits of free hosting tiers.
- Issues that need an already-compromised machine or administrator access to the host running the app.
- The **default demo PINs** and other items listed under *Known Limitations* below, unless you show a new way to exploit them.
- Forks, mirrors or copies of this project maintained by others.

## Known Limitations and Security Notes

We want to be upfront about the current state of this prototype:

- **Demo PINs:** the RBAC module ships with default demo PINs (LHW `1111`, Senior Radiologist `9999`) that are shown in the UI for demonstration. They provide **no real security**.
- **Local cache is not encrypted:** `pink_edge_cache.db` is a plain SQLite file. Anyone with access to the file system can read it.
- **Simulated cloud sync:** the Alibaba Cloud sync and its "synced" status are simulated. No data actually leaves the device through this path.
- **GSM alerts:** alert payloads include a patient identifier, location and BI-RADS category, and SMS is not an encrypted channel.
- **Anonymization claims:** the DICOM anonymizer is *designed to support* HIPAA/GDPR-style de-identification. It has **not** been independently audited or certified, and it should not be relied on as the only safeguard.
- **Cloud inference:** when a Roboflow key is configured, images are sent to Roboflow's hosted API over the internet for inference.
- **No transport security of its own:** the Streamlit server serves plain HTTP unless you put it behind a TLS-terminating reverse proxy.

## Guidance for Deployers

If you run Pink Edge AI anywhere other than your own machine for a demo:

1. **Do not use it with real patient data** until it has been independently validated, security-reviewed and approved under the regulations that apply to you.
2. **Change the default PINs** and do not show PINs in the UI. Load them from environment variables or secrets, not from source code.
3. **Keep API keys out of git.** Use the `ROBOFLOW_API_KEY` environment variable or Streamlit *App settings → Secrets*. Make sure `.gitignore` includes `roboflow_key.txt`, `.streamlit/secrets.toml`, `*.db`, `.env` and `__pycache__/`. [streamlit +2](https://discuss.streamlit.io/t/add-secrets-to-your-streamlit-apps/11738) If a key is ever committed or shared, **revoke and regenerate it straight away**; deleting the file is not enough.
4. **Protect the report cache.** Restrict file permissions on `pink_edge_cache.db`, keep it on an encrypted disk, never commit it, and delete it when you no longer need it.
5. **Do not expose the web edition publicly as-is.** Bind to `localhost`, or put it behind HTTPS and proper authentication before using `--server.address 0.0.0.0`.
6. **Keep dependencies up to date.** Pin versions, watch for security advisories (e.g., enable Dependabot alerts), and update promptly.
7. **Only load model weights from trusted sources**, and check their integrity where possible.
8. **Treat GSM alerts as sensitive.** Send them only to verified recipients and keep identifiers in them to a minimum.

## Safe Harbor

We support good-faith security research. If you:

- make a good-faith effort to follow this policy,
- avoid privacy violations, data destruction and service disruption,
- use only synthetic data or data you own, and never try to access real patient data,
- test only against your own local or self-hosted copy of Pink Edge AI, not against other people's deployments,
- stop and report straight away if you encounter sensitive data, and
- give us reasonable time to fix the issue before you disclose it publicly,

then we will consider your research authorised, we will not pursue or support legal action against you over it, and we will work with you to understand and fix the issue. This safe harbor covers only this project's maintainers. It does not cover third parties such as Roboflow, Hugging Face or hosting providers, which have their own policies.

## Acknowledgements

We are grateful to everyone who helps keep Pink Edge AI and its users safe. With your permission, we will credit reporters of valid vulnerabilities in the published GitHub Security Advisory and in this section.

*No vulnerabilities have been reported yet.*

---

*This policy may be updated as the project matures. Last updated: [date].*
````

## Recommendations (in order)

1. **Turn on private vulnerability reporting before you publish.** The "Report a vulnerability" link in the policy only works once a repository owner or administrator enables it under Settings → Advanced Security → Private vulnerability reporting. Then commit the file as `SECURITY.md` at the root. GitHub Docs say it can also go in "your repository's root, docs, or .github folder."
2. **Fill in the two placeholders.** `[maintainer email]` needs a real, monitored address, ideally a project alias rather than a personal address. `[date]` needs the date you publish. I found no email anywhere in the repo, so I did not make one up. If you prefer GitHub-only contact, delete the email sentence.
3. **Fix findings 1–4 in the same PR.** Untrack the cache DB and `__pycache__/`, harden `.gitignore`, and remove the on-screen PINs (or check that they really are on screen). Revoke any Roboflow key that has ever appeared in a public repo, including copies.
4. **Correct the documents.** Change "HIPAA/GDPR compliant" to "designed to support HIPAA/GDPR-style de-identification" in the README, and fix the stale `App/` paths. If the cache turns out to be encrypted, or the PINs are not shown on screen, edit those *Known Limitations* bullets to match.
5. **Turn on the free GitHub security features for public repos:** Dependabot alerts, secret scanning with push protection (GitHub Docs: "Secret Protection features are available for all public repositories"), and branch protection on `main`.

## Caveats

- I saw the repo through its GitHub front page and README (82 commits; file and folder listing; empty Releases section). [github](https://github.com/Zobia-Irshad/Pink_Edge_AI) The tool could not open individual files. So the default PINs, the `.gitignore` contents, dependency pinning, how PINs are stored, the hashing method, and the cache's contents and encryption **are not verified**. They come from the README, your prior analysis or inference, and the Key Findings table marks which is which. Check them by cloning the repo before you publish.
- I could not check the tags page. The policy assumes there are no versioned tags. If tags exist, add rows for them.
- The `roboflow_key.txt` finding is in `Faheem078/Pink_Edge_AI-main`, not in your repo. I could not tell whether it holds a real key or whose key it is. A third copy, `Danger-Khan/Medical-Radiology-AI`, also exists. [github](https://github.com/Faheem078/Pink_Edge_AI-main) [github](https://github.com/Danger-Khan/Medical-Radiology-AI) The policy marks all such copies as unsupported.
- The 5-day / 14-day / 90-day timelines are reasonable choices for a volunteer project, based on common practice. One reference point is Google Project Zero's 90-day disclosure window; its 2020 policy update reports that "97.7% of our vulnerability reports are fixed within our 90 day disclosure policy." Another is acknowledgement within 3–5 business days in many published disclosure policies. They are not a legal requirement. Promise only timelines you can actually meet.
- This repo is a demo, not a HIPAA-covered entity's system. If it is ever deployed with real data, legal breach-notification duties belong to the deploying organisation and are separate from this policy. For example, 45 CFR §164.404(b) requires notice "without unreasonable delay and in no case later than 60 calendar days after discovery of a breach."
