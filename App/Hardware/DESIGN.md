# Physical Design

Enclosure, power, and thermal design considerations for deploying the hardware in
`HARDWARE_REQUIREMENTS.md`/`WIRING.md` into an actual rural Basic Health Unit — a real clinic room,
not a lab bench. Written for the RK3588 build (`ALTERNATIVES.md` Option A); notes where the
Raspberry Pi build differs.

## Design constraints, from the project's own stated environment

Straight from this project's own framing (`../Documentations/PROJECT_ARCHITECTURE.md`): rural BHUs
in Punjab, unreliable power, unreliable connectivity, no on-site IT staff. That drives every design
decision below more than any aesthetic one.

## Enclosure

- **Ventilated, not sealed** — the RK3588's NPU generates real heat under sustained inference load;
  a fully sealed case without airflow will thermal-throttle within minutes of back-to-back triages.
  Minimum: intake + exhaust vents with dust mesh, positioned so the heatsink/fan (see
  `HARDWARE_REQUIREMENTS.md`) has a clear airflow path.
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
  failover — a simple diode-OR circuit or a purpose-built UPS HAT (see `HARDWARE_REQUIREMENTS.md`)
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
  `WIRING.md`) should also be on the battery-backed rail, so alerts can still go out during a power
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

- **Status at a glance**: the ESP32-driven status LEDs (`WIRING.md`) should be visible without
  opening the case — power / GSM-registered / alert-sent, in that order left-to-right, is the
  convention used elsewhere in this doc set.
- **No exposed electronics during normal use** — the SIM-swap compartment (above) is the only part
  that should ever need opening in the field; everything else is set-and-forget.
- **Touchscreen height/angle** if the optional display is fitted: desk-height viewing angle, not a
  server-rack vertical mount — this is a clinical tool an LHV will look at during a patient
  interaction, not a status board glanced at from across a room.

## What a first pilot unit can skip

Consistent with `HARDWARE_REQUIREMENTS.md`'s "not needed" section — a first pilot doesn't need a
custom-molded enclosure, a UPS HAT with graceful-shutdown scripting, or a split-compartment design.
An off-the-shelf ventilated project box, a basic 12V SLA + manual power switch, and taping the SIM
slot accessible is a legitimate v0 — the items above are what to add once the pilot proves the
concept and a second unit gets funded.
