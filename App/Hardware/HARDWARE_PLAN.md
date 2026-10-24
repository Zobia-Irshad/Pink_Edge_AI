# Pink Edge AI — Hardware Plan

Consolidated from 6 separate files (`ALTERNATIVES.md`, `ARCHITECTURE.md`, `DESIGN.md`,
`HARDWARE_REQUIREMENTS.md`, `STRUCTURE.md`, `WIRING.md`) — content unchanged, just combined into
one file. See `README.md` for the folder overview and `BOM.txt` for the flat procurement parts list
(kept separate — plain-text/tabular, a different format from the rest).

---

## Alternatives (hardware choice, and why)

# Hardware Alternatives — Comparison & Recommendation

Four options were on the table. They aren't actually competing for the same job — three are
candidate **main compute boards** (run the AI models), one is a **companion microcontroller** that
can't run the models at all but is useful alongside any of the other three. Laying that out first
avoids the apples-to-oranges trap of putting an MCU in a "which one wins" table with a full SBC.

## Quick verdict

- **Main compute: RK3588 SBC** (e.g. Orange Pi 5 / Radxa Rock 5B) — matches the project's original
  pitch, has a real NPU, genuinely fast for YOLOv8/RF-DETR-class models.
- **Budget/availability fallback: Raspberry Pi 4/5** — cheaper, everywhere in Pakistan, but no NPU
  — needs a USB AI accelerator (Coral / Hailo-8) to not be painfully slow, or accepts CPU-only
  inference at a real latency cost.
- **Zero-new-hardware fallback: the Android app** (already built — see `../GUI.py`'s Android
  discussion history) on LHVs' own phones. No procurement, no shipping, no import duties — the
  realistic answer for sites where a dedicated board just won't happen this budget cycle.
- **ESP32: not a compute alternative** — it's a companion MCU for the GSM/telemetry subsystem
  (see the Wiring section), freeing the main board's UART and adding a battery-backed watchdog. Pair it
  with whichever of the three above gets chosen; it doesn't replace any of them.

## Comparison table

| | RK3588 SBC | Raspberry Pi 4/5 | Mobile (Android APK) | ESP32 (companion role) |
|---|---|---|---|---|
| **Role** | Main compute | Main compute (budget) | Main compute (zero-hardware) | GSM/telemetry companion only |
| **AI inference** | Real NPU, ~6 TOPS INT8 | CPU-only (~slow) or +Coral/Hailo accelerator | Phone SoC (CPU, sometimes NPU on newer phones) | None — cannot run these models |
| **Typical unit cost (2026)** | $60–120 (board) | $45–80 (board) + $60–130 (Coral/Hailo) if added | $0 (existing phone) | $4–8 |
| **Power draw** | 5–12W under load | 3–7W (+accelerator draw) | Phone battery, charges normally | <1W |
| **Availability in Pakistan** | Import-order territory (AliExpress/Taobao resellers), lead time weeks | Widely stocked locally, same-day in Karachi/Lahore | Already owned by most LHVs | Widely available, cheap |
| **Offline capability** | Full — NPU inference is 100% local | Full, but slow without an accelerator | Full — see `../inference.py`'s offline paths | N/A (not doing inference) |
| **Ruggedization** | Needs a case + fan/heatsink, no built-in battery | Same — needs case + power bank | Already ruggedized (a phone), some are IP-rated | Trivial to enclose, tiny footprint |
| **Development effort already done** | This repo's inference stack targets RK3588 conceptually (see `../Documentations/TECHNICAL_REFERENCE.md`) | Same Python stack runs unmodified, just slower | Full offline Android build was scoped and deferred (see chat history / `../Assets/Changes/`) | New — GSM/AT-command firmware to write |
| **Best fit** | A funded pilot deployment at 1–2 flagship BHUs | Scaling out to more BHUs once budget allows | Immediate rollout with zero procurement | Any deployment that wants 2G GSM failover without tying up the main board's UART |

## Why not ESP32 as the main board

It's a microcontroller (Xtensa/RISC-V, no MMU, no real OS, hundreds of KB to a few MB of RAM). It
cannot load a YOLOv8/RF-DETR model, run PyTorch/ONNX Runtime, or hold a 512×512 image buffer
comfortably alongside a network stack. Framing it as a "competitor" to the RK3588/Pi would be
dishonest about what it can do — same posture this project already takes about model accuracy
(see `../Documentations/MODEL_SOURCES.md`). Its actual value here: reliable, low-power, always-on
control of the SIM800L GSM module and hardware status indicators, independent of whatever the main
board is doing (including while it's rebooting or under heavy inference load).

## Recommendation for this project, concretely

1. **Prototype/demo now**: keep using the desktop/Streamlit editions already built — no hardware
   blocker to keep developing software.
2. **First pilot hardware**: one RK3588 SBC (Orange Pi 5 4GB/8GB is the cheapest common option) +
   an ESP32 companion for GSM, at a single BHU, to validate the real edge-inference path end to end
   (today the app's "hardware diagnostics" are illustrative — see `../Documentations/TECHNICAL_REFERENCE.md` — this
   is the step that makes them real).
3. **Parallel, zero-cost path**: finish and distribute the Android build to LHVs who already have a
   capable phone, for sites where a board isn't funded yet.
4. **Scale-out**: Raspberry Pi + Coral USB Accelerator once there's a second pilot to fund, if
   RK3588 boards remain hard to source locally.

---

## Architecture

# System Architecture

How the hardware in the Hardware Requirements section fits together, and how it maps onto the software
already built in this repo (`../GUI.py`, `../streamlit_app.py`, `../inference.py`,
`../offline_cv.py`). See `diagrams/system_architecture.png` for the visual version of this page.

## Three tiers, same as the original pitch

```
Rural BHU (Edge Node)  --2G GSM / WiFi-->  Alibaba Cloud (optional)  --2G GSM-->  Allied Hospital Hub
```

This repo's software already models all three tiers in simulation
(`../Documentations/TECHNICAL_REFERENCE.md`); the hardware described here is what turns tier 1 (the edge node) from a
laptop demo into a real deployable box, and gives tier 1→3 a real (if minimal) transport instead of
an in-process mock.

## Tier 1 — Edge Node (the box that sits in the BHU)

Two independent subsystems talking over a single UART, deliberately decoupled so a GSM hiccup can't
freeze the AI pipeline and a slow inference run can't drop a GSM alert:

**Compute subsystem** (RK3588 SBC or Raspberry Pi, per the Alternatives section):
- Runs the exact software already in this repo — `GUI.py` (Tkinter, for a local touchscreen) or
  `streamlit_app.py` (browser-based, if the node has a network client available) — no code changes
  needed to run on real hardware, just a real Python + the same `requirements.txt`.
- `inference.py` + `offline_cv.py` do the actual triage — on an RK3588, model inference should
  eventually route through the NPU via RKNN-converted weights for real speed (today it runs the
  same CPU-path PyTorch/ONNX code as the desktop build; NPU conversion is a follow-up, not blocking
  a first pilot — CPU inference on an RK3588 is still usable, just not NPU-fast).
- `pink_edge_cache.db` (SQLite) persists locally on the SBC's storage — survives reboots and power
  cuts by design (that's the whole point of the offline-first cache).

**GSM/telemetry subsystem** (ESP32 companion, per the Wiring section):
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
the main board's UART — see the Wiring section for both options.

---

## Design (enclosure, power, thermal, human factors)

# Physical Design

Enclosure, power, and thermal design considerations for deploying the hardware in
the Hardware Requirements section/the Wiring section into an actual rural Basic Health Unit — a real clinic room,
not a lab bench. Written for the RK3588 build (the Alternatives section Option A); notes where the
Raspberry Pi build differs.

## Design constraints, from the project's own stated environment

Straight from this project's own framing (`../Documentations/TECHNICAL_REFERENCE.md`): rural BHUs
in Punjab, unreliable power, unreliable connectivity, no on-site IT staff. That drives every design
decision below more than any aesthetic one.

## Enclosure

- **Ventilated, not sealed** — the RK3588's NPU generates real heat under sustained inference load;
  a fully sealed case without airflow will thermal-throttle within minutes of back-to-back triages.
  Minimum: intake + exhaust vents with dust mesh, positioned so the heatsink/fan (see
  the Hardware Requirements section) has a clear airflow path.
- **IP54 or better** if the clinic room isn't climate-controlled — dust ingress is the realistic
  threat in rural Punjab more than water, but both deserve the gasket-and-vent-mesh treatment IP54
  implies.
- **Split-compartment layout recommended**: one compartment for the SBC + ESP32 + GSM module
  (electronics, needs airflow but not user access), one for the SIM card slot + status LEDs + power
  switch (needs occasional physical access, e.g. swapping a SIM) — avoids opening the electronics
  bay just to check a SIM.
- **Mounting**: wall-mountable bracket or a weighted desktop footprint — either works; wall-mount
  reduces accidental knocks/spills in a busy clinic room, desktop is easier to service.

## Power resilience

This is the highest-leverage design decision for a rural deployment, not an afterthought:

- **Input**: accept both mains (via the board's rated PSU) and a battery pack, with automatic
  failover — a simple diode-OR circuit or a purpose-built UPS HAT (see the Hardware Requirements section)
  both work; the UPS HAT is simpler to get right and worth the extra cost for a pilot.
- **Graceful shutdown, not just battery backup**: SBCs (especially running off a microSD/eMMC) are
  vulnerable to storage corruption on sudden power loss. A UPS HAT with a low-battery GPIO signal,
  paired with a systemd service that triggers a clean `shutdown` at (say) 15% remaining, is worth
  the software effort — losing the SQLite cache to a corrupted filesystem after a power cut would
  undermine the entire "offline-first, never lose data" premise this project is built on.
- **Sizing**: a 12V/7Ah SLA battery gives roughly 4–8 hours of edge-node runtime depending on
  inference frequency — size up if the BHU's power cuts regularly run longer than that; ask the
  actual site, don't guess.
- **GSM module power is a separate concern** from the above — its own regulated supply (see
  the Wiring section) should also be on the battery-backed rail, so alerts can still go out during a power
  cut, arguably the single most important moment for them to work.

## Thermal

- RK3588 boards throttle around 85°C junction temperature; sustained inference bursts in an
  unconditioned room in Punjab summer (ambient 40°C+) will get there without active cooling — the
  heatsink+fan in the BOM is not optional for this climate, even though some RK3588 boards ship
  "passive-cooling-capable" for lighter workloads.
- Mount the fan to exhaust hot air away from the GSM module and battery — both have their own heat
  sensitivity (SIM800L transmit bursts get warm; SLA/Li-ion batteries degrade faster if kept hot).
- Raspberry Pi variant: lighter thermal load than RK3588 under CPU-only inference (less silicon
  doing work, but also runs *inference itself* longer, so the total "case gets warm" duration per
  triage may be comparable) — a heatsink is still recommended, a fan less critical than for RK3588.

## Human factors (the LHV using this device, not an engineer)

- **Status at a glance**: the ESP32-driven status LEDs (the Wiring section) should be visible without
  opening the case — power / GSM-registered / alert-sent, in that order left-to-right, is the
  convention used elsewhere in this doc set.
- **No exposed electronics during normal use** — the SIM-swap compartment (above) is the only part
  that should ever need opening in the field; everything else is set-and-forget.
- **Touchscreen height/angle** if the optional display is fitted: desk-height viewing angle, not a
  server-rack vertical mount — this is a clinical tool an LHV will look at during a patient
  interaction, not a status board glanced at from across a room.

## What a first pilot unit can skip

Consistent with the Hardware Requirements section's "not needed" section — a first pilot doesn't need a
custom-molded enclosure, a UPS HAT with graceful-shutdown scripting, or a split-compartment design.
An off-the-shelf ventilated project box, a basic 12V SLA + manual power switch, and taping the SIM
slot accessible is a legitimate v0 — the items above are what to add once the pilot proves the
concept and a second unit gets funded.

---

## Hardware Requirements (what to buy)

# Hardware Requirements

What's actually needed to build a real Pink Edge AI edge node, per the option chosen in
the Alternatives section. Prices are rough 2026 street-price estimates (USD), for planning only — get real
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
| Companion MCU (optional but recommended) | ESP32 DevKitC | Offloads GSM AT-command handling from the main board — see the Alternatives section | $4–8 |
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

No new hardware — see the Alternatives section. Requirements are just:

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
| SIM800L breakout | With onboard 2A-capable regulator (raw modules need external 4V/2A supply — see the Wiring section) | 2G modem | $5–10 |
| Status LEDs (x2–3) | 5mm, with 220Ω resistors | Power / GSM-registered / alert-sent indicators | <$1 |
| Li-ion backup (optional) | 18650 cell + TP4056 charge module | Keeps GSM alerts working through brief main-power loss | $5–10 |

## Not needed / deliberately out of scope

- **No GPU**: none of these boards need a discrete GPU; the RK3588's NPU and the Coral/Hailo
  accelerators are purpose-built for exactly this INT8 inference workload and are both cheaper and
  more power-efficient than a GPU would be here.
- **No custom PCB for a first pilot**: everything above is off-the-shelf breakout boards and
  jumper/Dupont wiring (see the Wiring section) — a custom PCB is a scale-out optimization, not a
  pilot-stage requirement.

---

## Structure (this folder's layout, fleet organization)

# Structure

Two things live in this file that don't fit the Architecture section (data flow) or the Design section (physical
build): how this documentation set itself is organized, and how a **fleet** of edge nodes structures
once it's more than one BHU — a different question from how one node works internally.

## This folder

```
Hardware/
  README.md                  — start here: index + executive summary
  HARDWARE_PLAN.md            — everything below, combined into one file (was 6 separate docs:
                                   Alternatives, Hardware Requirements, Architecture, Wiring,
                                   Design, and this Structure section itself)
  BOM.txt                     — flat parts list, plain text, copy-pasteable for procurement
  IMAGE_PROMPTS.md            — copy-paste prompts for an image-generating LLM
  diagrams/
    system_architecture.png   — visual for the Architecture section
    wiring_rk3588.png          — visual for the Wiring section, Option A
    wiring_raspberry_pi.png    — visual for the Wiring section, Option C
    wiring_esp32_gsm.png       — visual for the Wiring section's ESP32 side
    hardware_comparison.png    — visual for the Alternatives section's comparison table
```

Read order for someone new to the project: `README.md` -> the Alternatives section (decide what to build)
-> the Hardware Requirements section + `BOM.txt` (buy it) -> the Architecture section (understand how it fits
together) -> the Wiring section (build it) -> the Design section (house it properly).

## Software-to-hardware component map

Every piece of hardware here exists to run something that's already built in this repo — nothing in
`Hardware/` invents new software requirements, it's purely "where does the existing stack run":

| Repo file | Runs on | Role |
|---|---|---|
| `../GUI.py` | Compute subsystem (SBC) | Local touchscreen UI, if the node has a display |
| `../streamlit_app.py` | Compute subsystem (SBC) | Browser UI — same node, or a client device on the same LAN |
| `../inference.py` | Compute subsystem (SBC) | Model dispatch: Roboflow (needs the GSM/WiFi uplink) -> `offline_cv.py` / offline HF models (fully local) |
| `../offline_cv.py` | Compute subsystem (SBC) | The no-internet-ever fallback — this is the one guaranteed to work with zero uplink |
| `pink_edge_cache.db` (SQLite) | Compute subsystem's local storage | Survives reboots/power cuts — see the Design section's power resilience section for why this matters |
| *(new, not yet written)* GSM AT-command firmware | ESP32 companion | Owns SIM800L, forwards alert payloads — see the Architecture section's GSM subsystem section |
| *(new, not yet written)* Alibaba Cloud SDK integration | Compute subsystem, GSM Failover mode only | Currently simulated in `../GUI.py` (`simulate_iot_sync()` etc.) — making this real is a software task once real cloud credentials exist, no additional hardware needed beyond the uplink already covered |

The GSM firmware is the one genuinely new piece of software this hardware plan implies — everything
else is "run the existing repo on real hardware instead of a desktop," which needs zero code
changes.

## Fleet structure (multiple BHUs)

Once past a single pilot node, the natural structure mirrors the project's own naming
(`../Documentations/TECHNICAL_REFERENCE.md`'s "Allied Hospital Faisalabad" hub receiving from
"Rural BHU" nodes):

```
                         Allied Hospital Hub (1x, urban)
                                    |
                    2G GSM / internet (many-to-one)
                    /            |             \
         BHU Node #001    BHU Node #002   ...  BHU Node #NNN
        (Rural, edge)    (Rural, edge)         (Rural, edge)
```

- **Node identity**: give each physical node a stable ID (`BHU-001`, `BHU-002`, ...) baked into its
  config, not derived from anything guessable (IMEI, MAC) — this ID is what ties its cached reports
  and GSM alerts back to a specific clinic location at the Hospital Hub.
- **Independence**: nodes don't talk to each other, only to the hub — a node going offline (power
  cut, GSM outage) has zero effect on any other node's operation. This is already the software's own
  assumption (`../GUI.py`'s per-session state, no node-to-node sync anywhere) — the hardware fleet
  structure just makes it explicit at the deployment level too.
- **Provisioning**: a new node should be flashable from a single base image (OS + this repo's
  software + dependencies pre-installed) with only the node ID and SIM card differing between
  units — avoids per-unit manual setup drift as the fleet grows past a handful of nodes.
- **Hub scaling**: one Hospital Hub instance can realistically monitor dozens of nodes before
  needing its own scale-out (multiple hub instances behind a shared database) — not a pilot-stage
  concern, noted here so it's not forgotten later.

---

## Wiring

# Wiring

Pin-level connections for each configuration in the Hardware Requirements section. See the matching
diagrams in `diagrams/` for the visual version of each section. Wire colors below are the
conventional ones (red=power, black=ground, yellow/green=signal) — not load-bearing, just easier to
follow with a multimeter in hand.

**Universal warning**: SIM800L draws current spikes up to ~2A during transmit bursts. Powering it
from a board's 3.3V/5V GPIO rail directly (instead of its own regulated supply) is the single most
common cause of "GSM module keeps browning out/resetting" reports for this exact chip — always give
it its own capacitor-backed supply, per the sections below.

## Option A — RK3588 SBC + ESP32 GSM companion (recommended)

**RK3588 SBC side:**
| SBC pin | Connects to | Notes |
|---|---|---|
| UART TX (e.g. GPIO 8 / UART1_TX) | ESP32 RX2 (GPIO 16) | Main board -> ESP32 alert payloads |
| UART RX (e.g. GPIO 9 / UART1_RX) | ESP32 TX2 (GPIO 17) | ESP32 -> main board status/ACK |
| GND | ESP32 GND | **Common ground — do this first**, before any signal wires |
| USB port | UVC camera | Or CSI connector if using a ribbon camera module |
| USB-C / barrel jack | 5V/4A PSU | Dedicated supply, not shared with peripherals |
| HDMI/DSI (optional) | Touchscreen | For local UI without a laptop |

**ESP32 side:**
| ESP32 pin | Connects to | Notes |
|---|---|---|
| TX2 (GPIO 17) | SIM800L RX | Through a voltage divider or level shifter if the SIM800L board is 3.3V-only logic but your ESP32 dev board runs its UART at 5V logic — check your specific breakout, many are already 3.3V-native |
| RX2 (GPIO 16) | SIM800L TX | |
| GND | SIM800L GND + shared system GND | Common ground across ESP32, SIM800L, and main SBC |
| 5V (VIN) | SIM800L VCC — **via its own 4V/2A regulated supply**, not the ESP32's 5V pin directly | See warning above |
| GPIO 2 | Status LED (GSM registered) + 220ohm resistor -> GND | |
| GPIO 4 | Status LED (alert sent) + 220ohm resistor -> GND | |
| 3V3 | Status LED (power) + 220ohm resistor -> GND | Always-on indicator |
| Micro-USB / USB-C | Dev PC (flashing firmware only) | Not used in normal operation |

**SIM800L side (in addition to the above):**
| SIM800L pin | Connects to |
|---|---|
| ANT | External 2G whip antenna (u.FL/SMA) |
| SIM slot | Local carrier SIM card |
| VCC | Dedicated 4V/2A regulated supply, **with a >=1000uF capacitor across VCC/GND at the module** to absorb TX current spikes |

## Option B — Direct wiring, no ESP32 (simplest first prototype)

Skip the companion MCU; wire SIM800L straight to the main board's secondary UART:

| SBC pin | Connects to |
|---|---|
| UART TX (secondary, e.g. `/dev/ttyS0` or `/dev/ttyAMA0`) | SIM800L RX |
| UART RX | SIM800L TX |
| GND | SIM800L GND (shared system ground) |
| — | SIM800L VCC: still its own dedicated 4V/2A supply, same warning as above — this doesn't change just because there's no ESP32 |

Software then talks to the modem directly via AT commands over that serial port — no firmware to
write, just a serial library call from Python. Simpler, but the main board now owns modem
housekeeping (retries, registration polling) on top of running inference — acceptable for a
prototype, revisit if it causes noticeable latency under real load.

## Option C — Raspberry Pi variant

Same wiring as Option A/B, with two differences:

1. **GPIO pin numbers differ** — Raspberry Pi's UART is GPIO 14 (TXD) / GPIO 15 (RXD) on the 40-pin
   header, 3.3V logic (matches SIM800L breakouts natively, no level shifting needed for the UART
   lines — SIM800L's VCC power line still needs its own supply exactly as above).
2. **AI accelerator, if used**: Coral USB Accelerator plugs into any USB 3.0 port — no wiring beyond
   that; Hailo-8 HAT mounts directly on the 40-pin header (check for GPIO pin conflicts with the
   UART pins above before stacking both).

## Camera module wiring (all SBC options)

- **USB UVC camera**: plug into any USB port, no additional wiring. Simplest option, works
  identically across RK3588 and Pi builds.
- **CSI ribbon camera** (if using the board's dedicated camera connector instead): follow the
  board's own CSI pinout exactly — ribbon orientation (contacts facing the board vs. away) differs
  between RK3588 boards and Raspberry Pi, check your specific board's documentation before
  connecting; inserting it backwards is a common (and usually harmless, if caught immediately)
  mistake.

## What NOT to do

- Don't power SIM800L from the SBC's or ESP32's onboard 5V/3.3V GPIO pins directly — under-current
  brownouts here are the #1 reported issue with this module across hobbyist forums.
- Don't skip the common-ground connection between boards before wiring signal lines — floating
  grounds between two independently-powered boards can damage both UARTs.
- Don't rely on the SIM800L's tiny onboard antenna trace for real deployment — always use the
  external whip antenna; rural signal is marginal enough already.
