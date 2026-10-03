#!/usr/bin/env bash
# One-shot setup for an RK3588 board (Debian/Ubuntu aarch64) acting as a Pink Edge AI edge node.
# Run on the board itself (not the dev machine): sudo ./install_rk3588.sh
set -euo pipefail

# EDIT ME: where to get App/ from. Left as a placeholder -- this is specific to your deployment
# (a git remote, an rsync source over the LAN, a USB drive mount, etc.), not something this script
# can assume. Two common examples are commented below.
PINKEDGE_APP_SRC="${PINKEDGE_APP_SRC:-}"
INSTALL_DIR="${HOME}/pinkedge"

if [ -z "$PINKEDGE_APP_SRC" ]; then
    echo "Set PINKEDGE_APP_SRC before running, e.g.:"
    echo "  PINKEDGE_APP_SRC=https://github.com/Danger-Khan/Medical-Radiology-AI.git ./install_rk3588.sh"
    echo "  PINKEDGE_APP_SRC=/media/usb/Pink-Edge-AI ./install_rk3588.sh   # copy from a USB drive"
    exit 1
fi

echo "== System packages =="
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git

echo "== Fetching App/ =="
mkdir -p "$INSTALL_DIR"
if [[ "$PINKEDGE_APP_SRC" == http*://*.git ]] || [[ "$PINKEDGE_APP_SRC" == git@* ]]; then
    git clone --depth 1 "$PINKEDGE_APP_SRC" "$INSTALL_DIR/repo"
    APP_DIR="$INSTALL_DIR/repo/App"
else
    cp -r "$PINKEDGE_APP_SRC" "$INSTALL_DIR/repo"
    APP_DIR="$INSTALL_DIR/App"
    [ -d "$INSTALL_DIR/repo/App" ] && APP_DIR="$INSTALL_DIR/repo/App"
fi
ln -sfn "$APP_DIR" "$INSTALL_DIR/App"

echo "== Python venv + deps (this step is slow -- torch on aarch64 is a big download) =="
python3 -m venv "$INSTALL_DIR/venv"
source "$INSTALL_DIR/venv/bin/activate"
pip install --upgrade pip
pip install -r "$APP_DIR/requirements.txt"
pip install -r "$(dirname "$0")/requirements_rk3588.txt"
deactivate

echo "== Enabling the UART (needed for uart_bridge.py, Option A in WIRING.md) =="
if command -v raspi-config >/dev/null 2>&1; then
    echo "Raspberry Pi detected instead of an RK3588 board -- use raspi-config to enable the UART."
else
    echo "On most RK3588 boards, enable the secondary UART via the board vendor's device-tree overlay"
    echo "tool (e.g. 'orangepi-config' on Orange Pi boards, 'rsetup' on Radxa boards) -- there is no"
    echo "single universal command across RK3588 vendors, so this step is manual. See your board's"
    echo "own wiki for '<board name> enable UART'."
fi

echo "== Installing the systemd kiosk service =="
SERVICE_SRC="$(dirname "$0")/pinkedge.service"
SERVICE_DST="$HOME/.config/systemd/user/pinkedge.service"
mkdir -p "$(dirname "$SERVICE_DST")"
sed "s#%h/pinkedge/App#$INSTALL_DIR/App#g; s#/usr/bin/python3#$INSTALL_DIR/venv/bin/python3#g" \
    "$SERVICE_SRC" > "$SERVICE_DST"
systemctl --user daemon-reload
systemctl --user enable pinkedge.service
echo "Installed. Reboot into the desktop session and pinkedge.service will autostart GUI.py."
echo "Check status any time with: systemctl --user status pinkedge.service"
