# Pink Edge AI — Privacy Policy

Author Name:  Imaad Ullah Khan
Author Email: yameenimaad@gmail.com
AI Helper:    Claude

**Last updated:** 2026-10-03 · Applies to: Pink Edge AI desktop (`GUI.py`) and web (`streamlit_app.py`) editions.

---

## 0. What this document is, and what it is not

This is the privacy policy for a **hackathon prototype**, not a deployed clinical product. It is
written to be checkable rather than reassuring: every claim below corresponds to something
`Tests/privacy_leak_test.py` actually verifies, and that suite's result is recorded in
`Evidence/EVIDENCE_REPORT.md`. Where the software does *not* protect something, this document says
so rather than staying quiet about it.

**No real patient data exists in this repository.** Every identifier the app handles today is
generated at random per session (`random.randint(10000000, 99999999)`). Nothing here has processed
a real person's scan. What follows therefore describes the *mechanism* — how the system would
handle identifiers if real ones were fed through it — and is honest about which parts of that
mechanism are implemented and which are not.

---

## 1. What data the app handles

| Category | Examples | Where it comes from |
|---|---|---|
| Medical image | mammogram, chest X-ray, fetal ultrasound | uploaded by the operator, or a bundled sample |
| Pseudonymous identifier | 8-digit patient ID, derived privacy hash | generated locally per session |
| Clinical metadata | age, modality, body part | entered/derived locally |
| Triage output | verdict, BI-RADS/severity, confidence, localization | produced by the model tiers |
| Operational telemetry | log lines, simulated hardware/network stats | generated locally, simulated |

The app does **not** ask for, and has no field for: patient name, CNIC/national ID, phone number,
address, date of birth, or GPS location.

---

## 2. Where data goes — every sink, and what protects it

### 2.1 On the device (at rest)

- **SQLite cache** (`App/pink_edge_cache.db`) — stores the triage report: pseudonymous patient ID,
  age, modality, verdict, BI-RADS/density, confidence, timestamp. The schema is verified to contain
  **no name, CNIC, phone, DOB, address, or coordinate column**.
- **Model weights and datasets** (`App/Models/`) — public research datasets and model checkpoints.
  No patient data.
- The cache is a plain local file. **It is not encrypted at rest.** On a shared or unattended
  machine, anyone with file access can read it. For a real deployment this would need full-disk
  encryption or an encrypted DB; that is not implemented here.

### 2.2 Leaving the device

- **GSM / IoT broadcast to the hospital hub** — carries `ID:<privacy hash>|LOC:ANON|<verdict code>`.
  The raw identifier is replaced by a SHA-256 hex privacy hash, and the location field is redacted
  to the literal `ANON`. **Both editions** do this; the suite checks each one separately, because
  the desktop edition previously did not (it broadcast the raw ID and a literal coordinate — found
  by `Tests/privacy_leak_test.py` and fixed).
- **Generated reports** (text / PDF) — contain the pseudonymous ID, age, and clinical findings.
  Verified to contain no name, CNIC, phone, DOB, or address. These files go wherever the operator
  saves them; **once exported, this app no longer controls them.**
- **Hosted model inference (Roboflow)** — *if* an API key is configured, the **image itself** is
  transmitted to Roboflow's servers for inference. This is the one path where clinical content
  leaves the device. It is optional, it is off unless a key is present, and every modality falls
  back to a fully local tier without it. See §4.
- **Alibaba Cloud sync** — **simulated only.** No credentials, no network calls, no data leaves the
  device on this path. The UI panel is a demonstration.

### 2.3 Never transmitted or stored

- The Roboflow API key is read from `App/roboflow_key.txt` (gitignored, verified untracked), an
  environment variable, or Streamlit secrets. It is verified to be absent from reports, the cache,
  log output, and all source files.

---

## 3. The privacy hash

`dicom_anonymizer.generate_hex_privacy_hash()` produces an uppercase SHA-256 hex digest over a
canonical (sorted-key) JSON encoding of the identifier bundle.

- **One-way** — verified that no input value, and no obvious fragment of one, is recoverable from
  the digest.
- **Deterministic** — the same patient hashes identically across sessions and across both editions,
  so a hub can correlate repeat scans without ever learning who the patient is.
- **Not salted.** This matters and is stated deliberately: an unsalted hash of a low-entropy
  identifier (an 8-digit ID has only 90 million possibilities) is **vulnerable to a brute-force
  reversal** by anyone who can guess the input format. For real deployment this needs a secret salt
  or an HMAC. It is adequate for a prototype handling synthetic IDs; it is **not** adequate for real
  patient identifiers.

`anonymize_dicom_metadata()` replaces the values of known-sensitive DICOM tags (`PatientName`,
`PatientID`, `PatientBirthDate`, `PatientAddress`, `PatientTelephoneNumbers`,
`ReferringPhysicianName`, `InstitutionName`) with per-tag hashes, while preserving clinically
necessary fields (age, sex, modality, body part) unchanged.

---

## 4. Offline by default; the cloud is opt-in

The app is designed to run with no internet, and this is verified rather than asserted:
`Validation/validate_offline.py` blocks every outbound socket, proves the block is real, and then
runs all three modalities through to a result.

- **With no API key**: nothing leaves the device. Inference runs on the offline pixel-diff
  heuristic, the locally-trained classifiers, and cached Hugging Face weights.
- **With an API key**: the uploaded image is sent to Roboflow for the hosted model tiers. If you do
  not want images leaving the device, **do not configure a key** — the app remains fully functional.

---

## 5. Access control

`auth_manager.py` provides two roles — Lady Health Worker (PIN 1111) and Senior Radiologist
(PIN 9999) — gating who may override an AI assessment.

**This is a UI demonstration, not a security control.** The PINs are hardcoded in source, there is
no account system, no session management, no audit trail of who changed what, and the permission
check fails closed but is trivially bypassed by anyone who can run the Python directly. It must not
be relied on to protect anything.

---

## 6. Retention and deletion

There is no automatic retention policy or expiry. Cached reports persist until deleted. To erase
all locally held data:

```
rm App/pink_edge_cache.db          # the report cache
```

Exported reports must be deleted wherever the operator saved them.

---

## 7. Regulatory posture

Pink Edge AI is **not** a medical device, is **not** FDA/CE/DRAP cleared, and has not undergone
clinical validation. It must not be used to make or defer a clinical decision. Its outputs are
illustrative, and several displayed fields are generated rather than measured — see
`Tests/fabrication_audit.py` and §3 of `Evidence/EVIDENCE_REPORT.md`, which identify exactly which.

Designing toward HIPAA/GDPR principles (data minimisation, pseudonymisation, local processing) is a
stated goal of the project. **Compliance with either regime has not been assessed or certified**, and
the gaps named in §2.1, §3, and §5 would each need to be closed first.

---

## 8. Verifying these claims yourself

```
cd App
python Tests/privacy_leak_test.py        # every claim in §2 and §3
python Validation/validate_offline.py    # every claim in §4
python Evidence/generate_evidence.py     # all suites + a signed, hashed record
```

---

## 9. Contact

Questions about this policy or the data handling described in it:
**Imaad Ullah Khan** — yameenimaad@gmail.com
