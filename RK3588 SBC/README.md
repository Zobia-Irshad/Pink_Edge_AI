# RK3588 SBC — Deployment Code

Code for turning an RK3588 single-board computer (Orange Pi 5/5B/5 Plus, Rock 5B, Radxa CM3, etc. —
any RK3588/RK3588S board running Debian/Ubuntu aarch64) into the real Tier-1 edge node described in
`../App/Hardware/ARCHITECTURE.md`. This folder holds the pieces that are specific to *this* board
and don't belong in the general cross-platform app code under `../App/`:

- Running `../App/GUI.py` unmodified as an autostarting kiosk app on the board's touchscreen.
- The UART bridge that hands alert payloads to the ESP32/SIM800L companion described in
  `../App/Hardware/WIRING.md` (Option A).
- Converting a trained YOLOv8 mammography checkpoint to `.rknn` and running it through the RK3588's
  NPU via `rknn-toolkit-lite2`, instead of CPU-only PyTorch/ONNX — the "follow-up, not blocking a
  first pilot" item `ARCHITECTURE.md` flags.

## Files

```
install_rk3588.sh        — one-shot setup script: system packages, Python venv, requirements,
                            enables the UART, copies the systemd service
requirements_rk3588.txt   — Python deps for this board (pyserial for the UART bridge; rknn-toolkit-lite2
                            is installed separately per Rockchip's own wheel — see NPU section below)
pinkedge.service          — systemd unit: autostarts `python GUI.py` in kiosk mode on boot,
                            restarts it if it crashes
uart_bridge.py            — reads alert lines from a local queue file and forwards them to the ESP32
                            over UART, in the exact `ID:<id>|LOC:<lat>|<sms>` format GUI.py already
                            builds (see integration note below); reads ACKs back
convert_to_rknn.py         — run on an x86 dev machine (NOT the board) with rknn-toolkit2: converts
                            an ONNX-exported YOLOv8 mammography model to .rknn for the NPU
rknn_infer.py              — runs ON the board: loads a converted .rknn model via rknn-toolkit-lite2
                            and returns the same result-dict shape as ../App/inference.py's
                            predict_mammography(), so it can be dropped into that dispatch chain
```

## 1. Base setup

```bash
scp -r "RK3588 SBC" pi@<board-ip>:~/pinkedge-rk3588   # copy this folder to the board
ssh pi@<board-ip>
cd ~/pinkedge-rk3588
chmod +x install_rk3588.sh
sudo ./install_rk3588.sh
```

`install_rk3588.sh` also clones/copies `App/` itself onto the board (edit the `PINKEDGE_APP_SRC`
variable at the top of the script to point at wherever you're syncing the main app code from — a
git remote, an rsync source, or a USB drive path — it's left as a placeholder since that's specific
to your deployment, not something this script can assume).

## 2. Kiosk autostart

`install_rk3588.sh` installs `pinkedge.service` to `/etc/systemd/user/` and enables it, so
`App/GUI.py` starts automatically on every boot, full-screen, on whatever display is attached
(HDMI/DSI touchscreen per `../App/Hardware/WIRING.md`). Check status with:

```bash
systemctl --user status pinkedge.service
journalctl --user -u pinkedge.service -f     # live log
```

## 3. UART bridge to the ESP32/SIM800L companion (GSM Failover mode)

`uart_bridge.py` owns `/dev/ttyS1` (or whatever `UART_PORT` you set — see `../App/Hardware/
WIRING.md`'s pin table) and does one job: watch a small queue file for new alert lines and send them
over serial to the ESP32, which then handles the actual SMS/GPRS send. This keeps the AI pipeline
and the GSM link decoupled exactly as `ARCHITECTURE.md` describes — a slow inference run can't block
a queued alert, and a GSM hiccup can't freeze the UI.

**Integration point** (a two-line addition to `App/GUI.py`, not made automatically by this folder —
review it yourself before enabling on a board that's actually wired up): wherever `sms_payload` is
built (`App/GUI.py`, the line building `f"ID:{self.pat_id}|LOC:29.344|{result['sms']}"`), append:
```python
with open(os.path.expanduser("~/pinkedge_alert_queue.txt"), "a", encoding="utf-8") as fh:
    fh.write(sms_payload + "\n")
```
Run the bridge alongside the app: `python uart_bridge.py` (or let `install_rk3588.sh` register it as
a second systemd unit — see the script's comments).

Run without any wiring: `python uart_bridge.py --dry-run` prints what it would send instead of
opening a serial port — useful for testing the queue-file mechanism before hardware is attached.

## 4. NPU-accelerated mammography inference (optional, follow-up)

CPU inference on an RK3588 already works today — `App/inference.py`'s existing Roboflow/offline-CV
paths need no changes to run on this board (it's just Linux + Python + the same `requirements.txt`).
This section is for when you have a **local** mammography `.pt`/`.onnx` checkpoint and want NPU
speed instead of CPU:

1. **On an x86 dev machine** (rknn-toolkit2's conversion tooling is x86-only, not something you run
   on the board itself): install `rknn-toolkit2` (`pip install rknn-toolkit2`, per
   [Rockchip's own repo](https://github.com/airockchip/rknn-toolkit2) — exact wheel depends on your
   Python version, check their releases page), export your YOLOv8 `.pt` to ONNX
   (`yolo export model=best.pt format=onnx opset=12`), then:
   ```bash
   python convert_to_rknn.py --onnx best.onnx --output mammography.rknn --platform rk3588
   ```
2. **Copy `mammography.rknn`** onto the board (e.g. into `~/pinkedge-rk3588/models/`).
3. **On the board**: install the board-matching `rknn-toolkit-lite2` runtime wheel (from the same
   Rockchip repo's releases — this one IS aarch64, install it on-device, not on the dev machine),
   then:
   ```bash
   python rknn_infer.py --model models/mammography.rknn --image path/to/scan.jpg
   ```
   `rknn_infer.py` prints the parsed result dict; wiring it into `App/inference.py`'s
   `predict_mammography()` as an earlier/parallel tier is a follow-up integration step, not done
   automatically here (it needs a real converted `.rknn` file to test against, which this
   environment doesn't have).

**Honesty note**: neither the RKNN conversion nor the NPU-accelerated inference above has been run
against a real converted model in this environment — no RK3588 hardware and no local trained
mammography `.pt` checkpoint are available here to test with. The scripts follow Rockchip's
documented `rknn-toolkit2`/`rknn-toolkit-lite2` API exactly, but treat them as a starting point to
validate on real hardware, not as pre-verified.
