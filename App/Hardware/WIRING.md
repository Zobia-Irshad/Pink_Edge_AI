# Wiring

Pin-level connections for each configuration in `HARDWARE_REQUIREMENTS.md`. See the matching
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
