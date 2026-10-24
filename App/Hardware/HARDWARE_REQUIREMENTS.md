# Hardware Requirements

What's actually needed to build a real Pink Edge AI edge node, per the option chosen in
`ALTERNATIVES.md`. Prices are rough 2026 street-price estimates (USD), for planning only — get real
quotes before procurement. See `BOM.txt` for a flat, copy-pasteable parts list.

## Option A — RK3588 SBC edge node (recommended pilot build)

| Component | Spec | Purpose | Est. cost |
|---|---|---|---|
| SBC | RK3588(S), 4–8GB RAM, e.g. Orange Pi 5 / Radxa Rock 5B | Main compute + 6 TOPS NPU for on-device inference | $60–120 |
| Storage | 64–128GB eMMC or UHS-I microSD (A2-rated) | OS + model weights + SQLite cache | $10–20 |
| Camera / scan input | USB UVC camera or CSI camera module (for live capture), or USB/SD card reader (for pre-existing DICOM/JPEG files from an ultrasound/X-ray machine) | Image acquisition | $10–40 |
| GSM module | SIM800L (2G, quad-band) | Telemetry uplink where there's no WiFi/broadband | $5–10 |
| GSM antenna | External 2G whip antenna, u.FL/SMA | Reliable signal in rural areas | $2–5 |
| SIM card | Local 2G-capable carrier SIM, data or SMS plan | Network access for the GSM module | Ongoing |
| Companion MCU (optional but recommended) | ESP32 DevKitC | Offloads GSM AT-command handling from the main board — see `ALTERNATIVES.md` | $4–8 |
| Power supply | 5V/4A USB-C PD or barrel-jack supply, rated for the SBC's peak draw | Stable power — RK3588 boards brown out on underrated supplies | $8–15 |
| Battery backup (optional) | UPS HAT or 12V/7Ah SLA + charge controller | Keeps the node up through rural power cuts | $15–40 |
| Cooling | Heatsink + 5V fan (active cooling — RK3588 throttles under sustained NPU load without one) | Thermal stability during inference bursts | $5–10 |
| Enclosure | IP54+ ventilated case, or 3D-printed/custom | Dust/humidity protection for a clinic environment | $10–30 |
| Display (optional) | 7–10" HDMI/DSI touchscreen | Local UI without a separate laptop | $30–60 |

**Est. total per node (with display, without battery backup): ~$150–280**

## Option B — Raspberry Pi edge node (budget/scale-out)

Same peripheral list as Option A (camera, GSM, power, enclosure), swap the SBC line for:

| Component | Spec | Purpose | Est. cost |
|---|---|---|---|
| SBC | Raspberry Pi 4 (4/8GB) or Pi 5 | Main compute (CPU-only NPU-less inference) | $45–80 |
| AI accelerator (strongly recommended) | Google Coral USB Accelerator (4 TOPS INT8) or Hailo-8 HAT (13 TOPS) | Without this, inference latency is multiple seconds to tens of seconds per image on CPU alone | $60–130 |

**Est. total per node (with accelerator, display, without battery backup): ~$180–320**

## Option C — Mobile (Android APK)

No new hardware — see `ALTERNATIVES.md`. Requirements are just:

| Requirement | Spec |
|---|---|
| Phone | Android 8.0+ (API 26+), 3GB+ RAM, any recent budget device |
| Storage | ~1–2GB free for model weights + app |
| Camera | Phone's own rear camera, for photographing printed/screen-displayed scans |
| Connectivity | Whatever the phone already has (2G/3G/4G/WiFi) for GSM-equivalent sync |

## GSM companion subsystem (ESP32) — add-on to Option A or B

| Component | Spec | Purpose | Est. cost |
|---|---|---|---|
| ESP32 DevKitC (or WROOM-32 bare module) | Dual-core, WiFi+BT, 30+ GPIO | Runs the GSM AT-command firmware independently of the main board | $4–8 |
| SIM800L breakout | With onboard 2A-capable regulator (raw modules need external 4V/2A supply — see `WIRING.md`) | 2G modem | $5–10 |
| Status LEDs (x2–3) | 5mm, with 220Ω resistors | Power / GSM-registered / alert-sent indicators | <$1 |
| Li-ion backup (optional) | 18650 cell + TP4056 charge module | Keeps GSM alerts working through brief main-power loss | $5–10 |

## Not needed / deliberately out of scope

- **No GPU**: none of these boards need a discrete GPU; the RK3588's NPU and the Coral/Hailo
  accelerators are purpose-built for exactly this INT8 inference workload and are both cheaper and
  more power-efficient than a GPU would be here.
- **No custom PCB for a first pilot**: everything above is off-the-shelf breakout boards and
  jumper/Dupont wiring (see `WIRING.md`) — a custom PCB is a scale-out optimization, not a
  pilot-stage requirement.
