# Structure

Two things live in this file that don't fit `ARCHITECTURE.md` (data flow) or `DESIGN.md` (physical
build): how this documentation set itself is organized, and how a **fleet** of edge nodes structures
once it's more than one BHU — a different question from how one node works internally.

## This folder

```
Hardware/
  README.md                  — start here: index + executive summary
  ALTERNATIVES.md             — RK3588 vs Raspberry Pi vs Mobile vs ESP32, comparison + recommendation
  HARDWARE_REQUIREMENTS.md    — bill of materials with specs, per option
  BOM.txt                     — flat parts list, plain text, copy-pasteable for procurement
  ARCHITECTURE.md             — system data flow, 3-tier model, software-to-hardware mapping
  WIRING.md                   — pin-level wiring per configuration
  DESIGN.md                   — enclosure, power resilience, thermal, human factors
  STRUCTURE.md                — this file
  diagrams/
    system_architecture.png   — visual for ARCHITECTURE.md
    wiring_rk3588.png          — visual for WIRING.md Option A
    wiring_raspberry_pi.png    — visual for WIRING.md Option C
    wiring_esp32_gsm.png       — visual for WIRING.md's ESP32 side
    hardware_comparison.png    — visual for ALTERNATIVES.md's comparison table
```

Read order for someone new to the project: `README.md` -> `ALTERNATIVES.md` (decide what to build)
-> `HARDWARE_REQUIREMENTS.md` + `BOM.txt` (buy it) -> `ARCHITECTURE.md` (understand how it fits
together) -> `WIRING.md` (build it) -> `DESIGN.md` (house it properly).

## Software-to-hardware component map

Every piece of hardware here exists to run something that's already built in this repo — nothing in
`Hardware/` invents new software requirements, it's purely "where does the existing stack run":

| Repo file | Runs on | Role |
|---|---|---|
| `../GUI.py` | Compute subsystem (SBC) | Local touchscreen UI, if the node has a display |
| `../streamlit_app.py` | Compute subsystem (SBC) | Browser UI — same node, or a client device on the same LAN |
| `../inference.py` | Compute subsystem (SBC) | Model dispatch: Roboflow (needs the GSM/WiFi uplink) -> `offline_cv.py` / offline HF models (fully local) |
| `../offline_cv.py` | Compute subsystem (SBC) | The no-internet-ever fallback — this is the one guaranteed to work with zero uplink |
| `pink_edge_cache.db` (SQLite) | Compute subsystem's local storage | Survives reboots/power cuts — see `DESIGN.md`'s power resilience section for why this matters |
| *(new, not yet written)* GSM AT-command firmware | ESP32 companion | Owns SIM800L, forwards alert payloads — see `ARCHITECTURE.md`'s GSM subsystem section |
| *(new, not yet written)* Alibaba Cloud SDK integration | Compute subsystem, GSM Failover mode only | Currently simulated in `../GUI.py` (`simulate_iot_sync()` etc.) — making this real is a software task once real cloud credentials exist, no additional hardware needed beyond the uplink already covered |

The GSM firmware is the one genuinely new piece of software this hardware plan implies — everything
else is "run the existing repo on real hardware instead of a desktop," which needs zero code
changes.

## Fleet structure (multiple BHUs)

Once past a single pilot node, the natural structure mirrors the project's own naming
(`../Documentations/PROJECT_ARCHITECTURE.md`'s "Allied Hospital Faisalabad" hub receiving from
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
