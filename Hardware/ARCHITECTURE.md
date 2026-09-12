# System Architecture

How the hardware in `HARDWARE_REQUIREMENTS.md` fits together, and how it maps onto the software
already built in this repo (`../GUI.py`, `../streamlit_app.py`, `../inference.py`,
`../offline_cv.py`). See `diagrams/system_architecture.png` for the visual version of this page.

## Three tiers, same as the original pitch

```
Rural BHU (Edge Node)  --2G GSM / WiFi-->  Alibaba Cloud (optional)  --2G GSM-->  Allied Hospital Hub
```

This repo's software already models all three tiers in simulation (`../Documentations/
PROJECT_ARCHITECTURE.md`); the hardware described here is what turns tier 1 (the edge node) from a
laptop demo into a real deployable box, and gives tier 1→3 a real (if minimal) transport instead of
an in-process mock.

## Tier 1 — Edge Node (the box that sits in the BHU)

Two independent subsystems talking over a single UART, deliberately decoupled so a GSM hiccup can't
freeze the AI pipeline and a slow inference run can't drop a GSM alert:

**Compute subsystem** (RK3588 SBC or Raspberry Pi, per `ALTERNATIVES.md`):
- Runs the exact software already in this repo — `GUI.py` (Tkinter, for a local touchscreen) or
  `streamlit_app.py` (browser-based, if the node has a network client available) — no code changes
  needed to run on real hardware, just a real Python + the same `requirements.txt`.
- `inference.py` + `offline_cv.py` do the actual triage — on an RK3588, model inference should
  eventually route through the NPU via RKNN-converted weights for real speed (today it runs the
  same CPU-path PyTorch/ONNX code as the desktop build; NPU conversion is a follow-up, not blocking
  a first pilot — CPU inference on an RK3588 is still usable, just not NPU-fast).
- `pink_edge_cache.db` (SQLite) persists locally on the SBC's storage — survives reboots and power
  cuts by design (that's the whole point of the offline-first cache).

**GSM/telemetry subsystem** (ESP32 companion, per `WIRING.md`):
- Owns the SIM800L modem exclusively — sends AT commands, watches for registration status, retries
  on failure — independent of whatever the compute subsystem is doing.
- Receives short alert payloads from the compute subsystem over UART (the same ~140-char strings
  `../GUI.py`'s `sms_payload` already constructs, e.g. `"ID:12345678|LOC:29.344|TB:POS"`), forwards
  them as SMS/GPRS to the Hospital Hub or cloud endpoint.
- Drives the status LEDs (power / GSM-registered / alert-sent) directly, so hardware status is
  visible even if the main board is rebooting.

## Tier 2 — Alibaba Cloud (optional, GSM Failover mode only)

Still exactly as simulated in the current software (`simulate_iot_sync()` / `simulate_oss_upload()`
/ `simulate_acr_check()` in `../GUI.py`) — IoT Platform for telemetry, OSS for high-risk image
backup, ACR for OTA model updates. Making this real (actual Alibaba Cloud SDK calls instead of an
in-process mock) is a software task, not a hardware one — no new hardware requirement here beyond
the GSM/WiFi uplink already covered in Tier 1.

## Tier 3 — Allied Hospital Hub (urban receiving terminal)

Doesn't need any of the specialized edge hardware — it's a receiving/monitoring station. A
Raspberry Pi, an old laptop, or a cloud VM running `streamlit_app.py` (or a future dedicated
receiver) all work; the requirement is just "always-on, always-connected," which most BHUs
themselves can't guarantee but an urban hospital generally can.

## Data flow, hardware-level

```
Scan (camera / SD card / manual upload)
  -> Compute subsystem: inference.py / offline_cv.py triage
  -> SQLite cache (local, survives power loss)
  -> [confirmed by LHV] -> UART -> ESP32
  -> ESP32 -> SIM800L -> 2G network -> Hospital Hub / Alibaba Cloud IoT
```

Every step up to and including the SQLite write works with **zero network connectivity** — that's
the "Fully Offline" mode the software already defaults to. The UART handoff to the GSM subsystem
only matters in "GSM Failover" mode, and even then, a GSM outage only delays the alert (it's queued,
not lost) rather than blocking triage.

## Why decouple compute and GSM onto two chips instead of one

The alternative — running the GSM modem's AT-command handling directly on the RK3588/Pi's own
UART — works too, and is simpler for a first prototype (one fewer component). The two-chip split
earns its complexity once you care about:
1. **Reliability**: an ESP32 pinging SIM800L status LEDs stays responsive even if the main board is
   mid-reboot after a power event — useful in an environment where a Rural BHU's own power supply
   is the least reliable link in the whole system.
2. **UART contention**: freeing the main board's primary UART for other peripherals (a
   barcode/DICOM scanner, a debug console) without needing a USB-to-serial adapter.
3. **Power sequencing**: the ESP32 can stay awake on a small battery even if the main compute board
   is deliberately powered down to save energy between triage sessions.

For a bare-minimum first prototype, it's reasonable to skip the ESP32 and wire SIM800L straight to
the main board's UART — see `WIRING.md` for both options.
