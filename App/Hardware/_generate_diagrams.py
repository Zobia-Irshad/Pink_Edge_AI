#!/usr/bin/env python3
"""
Generates the PNG diagrams in Hardware/diagrams/ from code — reproducible, editable, no binary
source-of-truth. Uses the exact color tokens from ../GUI.py's `C` dict for visual consistency with
the rest of the app. Run with:  python Hardware/_generate_diagrams.py
"""
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "diagrams")
os.makedirs(OUT, exist_ok=True)

# Same palette as GUI.py's `C` dict.
BG = "#0a0e1a"
SURFACE = "#111827"
SURFACE_ALT = "#1e293b"
BORDER = "#334155"
TEXT = "#f1f5f9"
TEXT_MUTED = "#94a3b8"
PRIMARY = "#14b8a6"
PRIMARY_LIGHT = "#2dd4bf"
ACCENT = "#ec4899"
SUCCESS = "#10b981"
WARNING = "#f59e0b"
DANGER = "#ef4444"


def font(size, bold=False):
    names = ["consolab.ttf", "arialbd.ttf"] if bold else ["consola.ttf", "arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


F_TITLE = font(30, bold=True)
F_H2 = font(20, bold=True)
F_BODY = font(15)
F_SMALL = font(13)
F_LABEL = font(13, bold=True)


def text_wh(draw, text, f):
    b = draw.textbbox((0, 0), text, font=f)
    return b[2] - b[0], b[3] - b[1]


def box(draw, xy, fill=SURFACE_ALT, outline=BORDER, width=2, radius=12):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def centered_text(draw, cx, y, text, f, color=TEXT):
    w, h = text_wh(draw, text, f)
    draw.text((cx - w / 2, y), text, font=f, fill=color)
    return h


def box_with_lines(draw, xy, title, lines, title_color=PRIMARY_LIGHT, fill=SURFACE_ALT, outline=BORDER):
    x0, y0, x1, y1 = xy
    box(draw, xy, fill=fill, outline=outline)
    cx = (x0 + x1) / 2
    y = y0 + 14
    y += centered_text(draw, cx, y, title, F_H2, title_color) + 10
    for ln in lines:
        y += centered_text(draw, cx, y, ln, F_SMALL, TEXT_MUTED) + 6
    return y


def h_arrow(draw, x0, x1, y, label="", color=PRIMARY):
    draw.line([(x0, y), (x1 - 12, y)], fill=color, width=3)
    draw.polygon([(x1, y), (x1 - 14, y - 7), (x1 - 14, y + 7)], fill=color)
    if label:
        w, h = text_wh(draw, label, F_SMALL)
        draw.rectangle([(x0 + x1) / 2 - w / 2 - 6, y - h - 14, (x0 + x1) / 2 + w / 2 + 6, y - 4], fill=BG)
        draw.text(((x0 + x1) / 2 - w / 2, y - h - 12), label, font=F_SMALL, fill=TEXT_MUTED)


def v_arrow(draw, x, y0, y1, label="", color=PRIMARY, label_side="right"):
    draw.line([(x, y0), (x, y1 - 12)], fill=color, width=3)
    draw.polygon([(x, y1), (x - 7, y1 - 14), (x + 7, y1 - 14)], fill=color)
    if label:
        dx = 10 if label_side == "right" else -10
        anchor = "la" if label_side == "right" else "ra"
        draw.text((x + dx, (y0 + y1) / 2 - 8), label, font=F_SMALL, fill=TEXT_MUTED, anchor=anchor)


def new_canvas(w, h, title, subtitle=None):
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    centered_text(d, w / 2, 24, title, F_TITLE, TEXT)
    if subtitle:
        centered_text(d, w / 2, 64, subtitle, F_BODY, TEXT_MUTED)
    return img, d


def footer(d, w, h, text):
    tw, th = text_wh(d, text, F_SMALL)
    d.text((w / 2 - tw / 2, h - 30), text, font=F_SMALL, fill=TEXT_MUTED)


# ============================================================
# 1. system_architecture.png
# ============================================================
def system_architecture():
    W, H = 1500, 830
    img, d = new_canvas(W, H, "Pink Edge AI — System Architecture", "Rural BHU Edge Node -> Alibaba Cloud (optional) -> Allied Hospital Hub")

    top = 130
    node_h = 560

    # Tier 1: Rural BHU Edge Node
    ex0, ex1 = 40, 470
    box(d, (ex0, top, ex1, top + node_h), fill=SURFACE, outline=PRIMARY, width=3)
    centered_text(d, (ex0 + ex1) / 2, top + 16, "RURAL BHU — EDGE NODE", F_H2, PRIMARY_LIGHT)

    box_with_lines(d, (ex0 + 20, top + 60, ex1 - 20, top + 260), "Compute Subsystem",
                    ["RK3588 SBC / Raspberry Pi", "GUI.py / streamlit_app.py", "inference.py + offline_cv.py",
                     "pink_edge_cache.db (SQLite)"], title_color=PRIMARY_LIGHT)
    d.text((ex0 + 20, top + 270), "UART", font=F_SMALL, fill=TEXT_MUTED)
    v_arrow(d, (ex0 + ex1) / 2, top + 262, top + 300, "alert payload", color=ACCENT)
    box_with_lines(d, (ex0 + 20, top + 310, ex1 - 20, top + 470), "GSM / Telemetry Subsystem",
                    ["ESP32 companion MCU", "SIM800L 2G modem", "Status LEDs"], title_color=ACCENT)
    centered_text(d, (ex0 + ex1) / 2, top + 490, "Power: mains + battery backup (see DESIGN.md)", F_SMALL, TEXT_MUTED)

    # Tier 2: Alibaba Cloud
    cx0, cx1 = 560, 960
    box(d, (cx0, top, cx1, top + node_h), fill=SURFACE, outline=WARNING, width=3)
    centered_text(d, (cx0 + cx1) / 2, top + 16, "ALIBABA CLOUD", F_H2, WARNING)
    centered_text(d, (cx0 + cx1) / 2, top + 46, "(optional — GSM Failover mode only)", F_SMALL, TEXT_MUTED)
    box_with_lines(d, (cx0 + 20, top + 90, cx1 - 20, top + 220), "IoT Platform", ["Telemetry ingest", "~140-char payloads"], title_color=WARNING)
    box_with_lines(d, (cx0 + 20, top + 240, cx1 - 20, top + 370), "OSS", ["High-risk image backup", "BI-RADS 4/5 patches only"], title_color=WARNING)
    box_with_lines(d, (cx0 + 20, top + 390, cx1 - 20, top + 500), "ACR", ["OTA model updates"], title_color=WARNING)

    # Tier 3: Hospital Hub
    hx0, hx1 = 1050, 1460
    box(d, (hx0, top, hx1, top + node_h), fill=SURFACE, outline=SUCCESS, width=3)
    centered_text(d, (hx0 + hx1) / 2, top + 16, "ALLIED HOSPITAL HUB", F_H2, SUCCESS)
    centered_text(d, (hx0 + hx1) / 2, top + 46, "(urban, always-connected)", F_SMALL, TEXT_MUTED)
    box_with_lines(d, (hx0 + 20, top + 90, hx1 - 20, top + 260), "Receiving Terminal",
                    ["Alert stream monitor", "streamlit_app.py client", "or dedicated receiver"], title_color=SUCCESS)
    box_with_lines(d, (hx0 + 20, top + 280, hx1 - 20, top + 420), "No specialized hardware needed",
                    ["Any always-on, always-connected", "device — Pi, laptop, or cloud VM"], title_color=SUCCESS)

    mid_y = top + node_h / 2
    h_arrow(d, ex1, cx0, mid_y - 60, "2G GSM / WiFi", PRIMARY)
    h_arrow(d, cx1, hx0, mid_y - 60, "2G GSM", WARNING)

    # Direct edge-node -> hub path (Fully Offline mode) routes BELOW both tier boxes, not
    # through the cloud box, so it reads as bypassing the cloud rather than passing through it.
    direct_y = top + node_h + 50
    d.line([(ex1, top + node_h - 30), (ex1, direct_y)], fill=ACCENT, width=3)
    h_arrow(d, ex1, hx0, direct_y, "", ACCENT)
    d.line([(hx0, direct_y), (hx0, top + node_h - 30)], fill=ACCENT, width=3)
    d.polygon([(hx0, top + node_h - 30), (hx0 - 7, top + node_h - 16), (hx0 + 7, top + node_h - 16)], fill=ACCENT)
    centered_text(d, (ex1 + hx0) / 2, direct_y + 10, "2G GSM — direct, Fully Offline mode (bypasses the cloud tier)", F_SMALL, ACCENT)

    footer(d, W, H, "Fully Offline mode uses the direct edge-node -> hub path only; GSM Failover mode adds the Alibaba Cloud hop.")
    img.save(os.path.join(OUT, "system_architecture.png"))


# ============================================================
# 2 & 3. wiring diagrams (RK3588 / Raspberry Pi)
# ============================================================
def wiring_diagram(filename, board_name, board_lines, uart_pins, extra_note):
    W, H = 1500, 1000
    img, d = new_canvas(W, H, f"Pink Edge AI — Wiring: {board_name} + ESP32 GSM Companion")

    top = 110
    # Main SBC box
    sx0, sx1, sy0, sy1 = 60, 560, top, top + 260
    box_with_lines(d, (sx0, sy0, sx1, sy1), board_name, board_lines, title_color=PRIMARY_LIGHT)

    # ESP32 box
    ex0, ex1, ey0, ey1 = 60, 560, sy1 + 140, sy1 + 380
    box_with_lines(d, (ex0, ey0, ex1, ey1), "ESP32 DevKitC", ["GSM companion MCU", "AT-command firmware"], title_color=ACCENT)

    v_arrow(d, (sx0 + sx1) / 2 - 60, sy1, ey0, uart_pins[0], label_side="left")
    v_arrow(d, (sx0 + sx1) / 2 + 60, ey0, sy1, uart_pins[1], color=SUCCESS, label_side="right")
    centered_text(d, (sx0 + sx1) / 2, sy1 + 6, "GND — common ground (wire first)", F_SMALL, DANGER)

    # SIM800L box
    gx0, gx1, gy0, gy1 = 620, 1120, ey0, ey1
    box_with_lines(d, (gx0, gy0, gx1, gy1), "SIM800L GSM Module",
                    ["TX/RX <-> ESP32 UART2", "VCC: dedicated 4V/2A supply", "+ >=1000uF capacitor", "ANT: external whip antenna"],
                    title_color=WARNING)
    h_arrow(d, ex1, gx0, (ey0 + ey1) / 2 - 30, "UART TX/RX", SUCCESS)
    h_arrow(d, gx0, ex1, (ey0 + ey1) / 2 + 40, "shared GND", DANGER)

    # Power box for SIM800L
    px0, px1, py0, py1 = 620, 1120, gy1 + 40, gy1 + 160
    box_with_lines(d, (px0, py0, px1, py1), "Dedicated 4V/2A Regulated Supply", ["NOT the ESP32 5V pin — see WIRING.md warning"], title_color=DANGER)
    v_arrow(d, (px0 + px1) / 2, py0 - 40, py0, "", color=DANGER)

    # LEDs + battery
    lx0, lx1, ly0, ly1 = 620, 1120, top, ey0 - 40
    box_with_lines(d, (lx0, ly0, lx1, ly1), "Status Indicators + Backup",
                    ["3x LED: power / GSM-registered / alert-sent", "(optional) 18650 + TP4056 battery backup"],
                    title_color=PRIMARY_LIGHT)
    v_arrow(d, (lx0 + lx1) / 2, ly1, ey0, "GPIO", color=PRIMARY)

    # Camera / power / display for main board
    px2 = 620
    box_with_lines(d, (px2, sy0, 1120, sy0 + 100), "Peripherals", ["USB/CSI Camera", "5V/4A PSU", "(optional) HDMI/DSI touchscreen"],
                    title_color=TEXT_MUTED)
    h_arrow(d, sx1, px2, sy0 + 50, "", PRIMARY)

    footer(d, W, H, extra_note)
    img.save(os.path.join(OUT, filename))


def wiring_rk3588():
    wiring_diagram(
        "wiring_rk3588.png", "RK3588 SBC (Orange Pi 5 / Rock 5B)",
        ["6 TOPS NPU", "Runs GUI.py / inference.py", "UART1 for ESP32 link"],
        ["UART1_TX -> RX2", "TX2 -> UART1_RX"],
        "See Hardware/WIRING.md Option A for the full pin table. GND must be common across all three boards before any signal wiring.",
    )


def wiring_raspberry_pi():
    wiring_diagram(
        "wiring_raspberry_pi.png", "Raspberry Pi 4/5",
        ["CPU-only (+ optional Coral/Hailo accelerator)", "Runs GUI.py / inference.py", "GPIO14/15 UART for ESP32 link"],
        ["GPIO14 (TXD) -> RX2", "TX2 -> GPIO15 (RXD)"],
        "See Hardware/WIRING.md Option C. Coral USB Accelerator: any USB3 port. Hailo-8 HAT: 40-pin header — check for GPIO conflicts with UART.",
    )


# ============================================================
# 4. wiring_esp32_gsm.png — zoomed detail
# ============================================================
def wiring_esp32_gsm():
    W, H = 1400, 850
    img, d = new_canvas(W, H, "Pink Edge AI — ESP32 <-> SIM800L Detail Wiring")

    top = 130
    ex0, ex1, ey0, ey1 = 100, 560, top, top + 420
    box(d, (ex0, ey0, ex1, ey1), fill=SURFACE, outline=ACCENT, width=3)
    centered_text(d, (ex0 + ex1) / 2, ey0 + 16, "ESP32 DevKitC", F_H2, ACCENT)
    pins = [("TX2 (GPIO17)", "-> SIM800L RX"), ("RX2 (GPIO16)", "<- SIM800L TX"),
            ("GND", "-> SIM800L GND (shared)"), ("GPIO 2", "-> LED: GSM registered"),
            ("GPIO 4", "-> LED: alert sent"), ("3V3", "-> LED: power (always-on)")]
    y = ey0 + 70
    for pin, desc in pins:
        d.text((ex0 + 24, y), pin, font=F_LABEL, fill=PRIMARY_LIGHT)
        d.text((ex0 + 24, y + 20), desc, font=F_SMALL, fill=TEXT_MUTED)
        y += 55

    gx0, gx1, gy0, gy1 = 800, 1300, top, top + 260
    box(d, (gx0, gy0, gx1, gy1), fill=SURFACE, outline=WARNING, width=3)
    centered_text(d, (gx0 + gx1) / 2, gy0 + 16, "SIM800L Module", F_H2, WARNING)
    for i, ln in enumerate(["RX <- ESP32 TX2", "TX -> ESP32 RX2", "GND -- shared ground", "VCC <- dedicated 4V/2A only",
                             "ANT -> external whip antenna"]):
        d.text((gx0 + 24, gy0 + 60 + i * 32), ln, font=F_SMALL, fill=TEXT_MUTED)

    h_arrow(d, ex1, gx0, (ey0 + ey1) / 2 - 100, "UART", SUCCESS)

    # power supply block
    sx0, sx1, sy0, sy1 = 800, 1300, gy1 + 60, gy1 + 220
    box(d, (sx0, sy0, sx1, sy1), fill=SURFACE, outline=DANGER, width=3)
    centered_text(d, (sx0 + sx1) / 2, sy0 + 14, "Dedicated 4V / 2A Regulator", F_H2, DANGER)
    centered_text(d, (sx0 + sx1) / 2, sy0 + 50, "+ >=1000uF capacitor across VCC/GND", F_SMALL, TEXT_MUTED)
    centered_text(d, (sx0 + sx1) / 2, sy0 + 75, "(absorbs TX current spikes up to ~2A)", F_SMALL, TEXT_MUTED)
    v_arrow(d, (sx0 + sx1) / 2, sy0 - 60, sy0, "VCC", color=DANGER)

    # battery block
    bx0, bx1, by0, by1 = 100, 560, ey1 + 60, ey1 + 220
    box(d, (bx0, by0, bx1, by1), fill=SURFACE, outline=SUCCESS, width=3)
    centered_text(d, (bx0 + bx1) / 2, by0 + 14, "Optional Battery Backup", F_H2, SUCCESS)
    centered_text(d, (bx0 + bx1) / 2, by0 + 50, "18650 Li-ion + TP4056 charge module", F_SMALL, TEXT_MUTED)
    v_arrow(d, (bx0 + bx1) / 2, ey1, by0, "5V", color=SUCCESS)

    footer(d, W, H, "NEVER power SIM800L VCC from the ESP32's own 5V/3V3 pins — see WIRING.md warning.")
    img.save(os.path.join(OUT, "wiring_esp32_gsm.png"))


# ============================================================
# 5. hardware_comparison.png
# ============================================================
def hardware_comparison():
    W, H = 1550, 900
    img, d = new_canvas(W, H, "Pink Edge AI — Hardware Options Compared")

    cols = ["Criteria", "RK3588 SBC", "Raspberry Pi", "Mobile (APK)", "ESP32 (companion)"]
    rows = [
        ("Role", "Main compute", "Main compute (budget)", "Main compute (zero-HW)", "GSM companion only"),
        ("AI inference", "Real NPU, ~6 TOPS", "CPU / +accelerator", "Phone SoC", "None"),
        ("Est. cost", "$60-120", "$45-80 (+accel.)", "$0 (existing phone)", "$4-8"),
        ("Power draw", "5-12W", "3-7W (+accel.)", "Phone battery", "<1W"),
        ("PK availability", "Import, weeks lead time", "Widely stocked locally", "Already owned", "Widely available"),
        ("Offline capable", "Yes — full NPU", "Yes, slower w/o accel.", "Yes — offline_cv.py etc.", "N/A"),
        ("Best fit", "Funded pilot, 1-2 BHUs", "Scale-out once funded", "Immediate rollout", "Any SBC build"),
    ]

    left = 40
    col_w = [260, 300, 300, 300, 290]
    top = 120
    row_h = 90
    x = left
    header_colors = [TEXT_MUTED, PRIMARY_LIGHT, PRIMARY_LIGHT, ACCENT, WARNING]
    for i, (c, w) in enumerate(zip(cols, col_w)):
        box(d, (x, top, x + w - 8, top + 50), fill=SURFACE, outline=BORDER, radius=8)
        centered_text(d, x + (w - 8) / 2, top + 14, c, F_LABEL, header_colors[i])
        x += w

    y = top + 58
    for r_i, row in enumerate(rows):
        x = left
        for c_i, (cell, w) in enumerate(zip(row, col_w)):
            fill = SURFACE_ALT if r_i % 2 == 0 else SURFACE
            outline = BORDER
            box(d, (x, y, x + w - 8, y + row_h - 8), fill=fill, outline=outline, radius=8)
            tcolor = PRIMARY_LIGHT if c_i == 0 else TEXT
            # wrap long cell text into up to 2 lines
            words, lines, cur = cell.split(" "), [], ""
            for word in words:
                trial = (cur + " " + word).strip()
                if text_wh(d, trial, F_SMALL)[0] > w - 30 and cur:
                    lines.append(cur)
                    cur = word
                else:
                    cur = trial
            if cur:
                lines.append(cur)
            ty = y + (row_h - 8 - len(lines) * 18) / 2
            for ln in lines:
                centered_text(d, x + (w - 8) / 2, ty, ln, F_SMALL if c_i else F_LABEL, tcolor)
                ty += 18
            x += w
        y += row_h

    footer(d, W, H, "Full comparison with rationale: Hardware/ALTERNATIVES.md")
    img.save(os.path.join(OUT, "hardware_comparison.png"))


if __name__ == "__main__":
    system_architecture()
    wiring_rk3588()
    wiring_raspberry_pi()
    wiring_esp32_gsm()
    hardware_comparison()
    print("Generated 5 diagrams in", OUT)
