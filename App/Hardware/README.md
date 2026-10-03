# Pink Edge AI — Hardware

The software in this repo (`../GUI.py`, `../streamlit_app.py`, `../inference.py`, `../offline_cv.py`)
runs today on a desktop or in a browser. This folder plans what it takes to run the same software
on real, deployable edge hardware in a rural Basic Health Unit — the "Edge Node" the project's own
architecture doc (`../Documentations/PROJECT_ARCHITECTURE.md`) already describes conceptually, made
concrete.

## Executive summary

- **Recommended pilot build**: an **RK3588 SBC** (e.g. Orange Pi 5 / Radxa Rock 5B) as main compute
  + an **ESP32** as a companion microcontroller dedicated to the GSM/telemetry link — see
  `ALTERNATIVES.md` for why, and the full parts list in `HARDWARE_REQUIREMENTS.md` / `BOM.txt`.
- **Two credible fallbacks**, not runners-up so much as different deployment situations: a
  **Raspberry Pi** build (cheaper, far more available in Pakistan, needs an AI accelerator to not be
  slow) for scaling out once a pilot is funded, and the **Android app** (zero new hardware — runs on
  LHVs' own phones) for immediate rollout where procurement isn't happening this cycle.
- **ESP32 is not a fourth compute option** — it can't run these models. It's a companion chip that
  pairs with either SBC option to own the SIM800L GSM modem independently of the main board. See
  `ALTERNATIVES.md`'s "Why not ESP32 as the main board" section.

## What's in this folder

| File | Answers |
|---|---|
| [`ALTERNATIVES.md`](ALTERNATIVES.md) | Which hardware, and why — full comparison table |
| [`HARDWARE_REQUIREMENTS.md`](HARDWARE_REQUIREMENTS.md) | What to buy, with specs and cost estimates |
| [`BOM.txt`](BOM.txt) | Flat parts list, plain text, for procurement |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | How the pieces fit together and talk to each other; how it maps onto the existing software |
| [`WIRING.md`](WIRING.md) | Exact pin connections to actually build it |
| [`DESIGN.md`](DESIGN.md) | Enclosure, power resilience, thermal, human factors for a real clinic deployment |
| [`STRUCTURE.md`](STRUCTURE.md) | This folder's own layout, and how a multi-BHU fleet is organized |
| [`diagrams/`](diagrams/) | The visual version of the above — architecture, wiring, and comparison diagrams (generated, schematic-accurate) |
| [`IMAGE_PROMPTS.md`](IMAGE_PROMPTS.md) | Copy-paste prompts for an image-generating LLM — realistic/marketing-style renders, not technical diagrams |

Start with `ALTERNATIVES.md` if you're deciding what to build, `WIRING.md`/`diagrams/` if you're
building the pilot unit that's already been decided on.

## Honesty note, consistent with the rest of this project

Nothing in this folder has been physically built or bench-tested — it's a hardware plan, written
with the same "state what's real vs. what's a recommendation" posture as
`../Documentations/MODEL_SOURCES.md` takes for the AI models. Component choices and wiring are
based on each part's published specs and common, well-documented usage patterns (SIM800L's power
requirements in particular are a widely-reported gotcha, not a guess) — but treat this as a build
plan to validate on a bench before field deployment, not as a pre-verified design.
