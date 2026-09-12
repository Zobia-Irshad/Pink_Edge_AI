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
  (see `WIRING.md`), freeing the main board's UART and adding a battery-backed watchdog. Pair it
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
| **Development effort already done** | This repo's inference stack targets RK3588 conceptually (see `../Documentations/PROJECT_ARCHITECTURE.md`) | Same Python stack runs unmodified, just slower | Full offline Android build was scoped and deferred (see chat history / `../Assets/Changes/`) | New — GSM/AT-command firmware to write |
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
   (today the app's "hardware diagnostics" are illustrative — see `PROJECT_ARCHITECTURE.md` — this
   is the step that makes them real).
3. **Parallel, zero-cost path**: finish and distribute the Android build to LHVs who already have a
   capable phone, for sites where a board isn't funded yet.
4. **Scale-out**: Raspberry Pi + Coral USB Accelerator once there's a second pilot to fund, if
   RK3588 boards remain hard to source locally.
