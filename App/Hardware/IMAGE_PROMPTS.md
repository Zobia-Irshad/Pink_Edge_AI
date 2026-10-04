# Image Prompts — for an image-generating LLM/tool

The diagrams in `diagrams/` are schematics I generated from code (`_generate_diagrams.py`) — clean
and accurate for wiring/architecture, but not photorealistic. If you want realistic product/render
-style images (for a pitch deck, README hero image, etc.), copy any prompt below into an
image-capable tool (Gemini, ChatGPT/DALL-E, Midjourney, Stable Diffusion). Each is self-contained —
paste one at a time, don't concatenate them.

General guidance: for Midjourney, append `--ar 16:9 --style raw` (or `--ar 1:1` for square); for
DALL-E/Gemini, the prose prompt alone is usually enough. Swap "photorealistic product photography"
for "clean 3D render, studio lighting" if you want a CGI look instead of a photo look.

---

## 1. RK3588 SBC — standalone board

```
Photorealistic product photography of a single-board computer (similar to an Orange Pi 5 or
Radxa Rock 5B), top-down 3/4 angle. Green PCB, RK3588 SoC chip visible in the center under a
small heatsink, GPIO pin header along one edge, USB-A ports, USB-C power port, HDMI port,
Ethernet jack, and a microSD card slot visible along the board edges. Studio lighting on a
neutral dark gray background, sharp focus, high detail, no hands or other objects in frame,
professional tech-product catalog style.
```

## 2. RK3588 edge node — full prototype assembly

```
Photorealistic photo of an electronics prototype on a wooden workbench: a green single-board
computer (RK3588-based SBC) connected via jumper wires to a small blue ESP32 development board,
which is in turn wired to a compact GSM modem module with a small black whip antenna. A USB
webcam sits nearby connected to the main board. Visible red and black power wires, a small
breadboard with status LEDs (one red, one green, one amber, lit). Warm workbench lighting,
shallow depth of field focused on the boards, slightly messy realistic wiring, engineering
project aesthetic, not overly clean or staged.
```

## 3. Finished enclosed device — clinic-ready unit

```
Photorealistic product photo of a small ventilated electronics enclosure, matte dark gray/black
plastic project box roughly the size of a paperback book, with a visible perforated vent grille
on one side, three small status LED indicators on the front panel (labeled with tiny icons:
power, signal, alert), a short stub antenna protruding from the top, and a USB-C power cable
exiting one side. Sitting on a plain clinic desk next to a stethoscope, softly blurred medical
background, natural daylight through a window, documentary/product-in-context photography style.
```

## 4. Wiring / breadboard close-up

```
Macro photorealistic photo of a breadboard prototyping setup: a small blue ESP32 development
board and a compact GSM modem module (SIM800L-style, with a small onboard antenna connector)
connected by four colored jumper wires (red, black, yellow, green) in a clean, organized
breadboard layout. Extreme close-up, shallow depth of field, sharp focus on the wire
connections and pin headers, soft diffused lighting, electronics engineering documentation
photography style.
```

## 5. Rural deployment context

```
Photorealistic wide-angle photo of the interior of a small rural health clinic room in Punjab,
Pakistan — simple whitewashed walls, a wooden desk, a health worker's chair, natural light from
a window with a curtain. On the desk sits a small ventilated black electronics enclosure with a
short antenna and a compact touchscreen tablet propped beside it showing a simple medical
dashboard UI. Warm, documentary photojournalism style, realistic and unglamorous, no visible
people, mid-afternoon lighting.
```

## 6. Hardware options side-by-side (for a pitch deck slide)

```
Clean product photography flat-lay, top-down view, four items arranged in a horizontal row on a
plain light gray background with soft even studio lighting: (1) a green RK3588 single-board
computer, (2) a black Raspberry Pi board with a red logo accent, (3) a modern Android
smartphone showing a simple app icon on its home screen, (4) a small blue ESP32 development
board. Equal spacing, consistent lighting and shadow direction across all four items, minimal
and professional, suitable for a technology comparison slide.
```

## 7. System architecture — stylized/artistic version (pitch deck hero image)

```
Modern flat-design technical illustration, dark navy background (#0a0e1a), showing three
connected nodes left to right: a small rural clinic building icon labeled "Edge Node" on the
left, a stylized cloud icon labeled "Cloud Sync" in the middle, and a hospital building icon
labeled "Hospital Hub" on the right, connected by glowing teal (#14b8a6) and pink (#ec4899)
signal-wave arrows. Clean vector illustration style, subtle glow effects, minimal geometric
shapes, professional tech-startup pitch-deck aesthetic, 16:9 composition.
```

---

**Note on accuracy**: these are for visual/marketing purposes (README hero images, pitch decks).
For anything that needs to be *technically correct* — actual pin connections, actual component
placement — use the generated diagrams in `diagrams/` instead; an image model has no knowledge of
this project's specific wiring and will invent plausible-looking but unverified connections.
