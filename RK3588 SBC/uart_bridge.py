#!/usr/bin/env python3
"""
Compute-subsystem side of the UART link to the ESP32/SIM800L companion described in
../App/Hardware/WIRING.md (Option A) and ../App/Hardware/ARCHITECTURE.md's "Tier 1" section.

Job: watch a small local queue file for new alert lines (one per triage that GUI.py appends to —
see the integration snippet in README.md) and forward each one to the ESP32 over UART, exactly as
`f"ID:{pat_id}|LOC:29.344|{result['sms']}"` already builds it in App/GUI.py. Reads back a one-line
ACK/status per send so a GSM outage is visible here without blocking the next queued alert.

Deliberately its own process (not imported into GUI.py) so a serial hiccup can't affect the UI
thread and vice versa -- matches the "decouple compute and GSM" reasoning in ARCHITECTURE.md.

Usage:
    python uart_bridge.py                      # real serial link
    python uart_bridge.py --dry-run            # print instead of opening a port (no hardware needed)
    python uart_bridge.py --port /dev/ttyS1 --baud 115200 --queue ~/pinkedge_alert_queue.txt
"""
import argparse
import os
import sys
import time

DEFAULT_QUEUE = os.path.expanduser("~/pinkedge_alert_queue.txt")
DEFAULT_PORT = "/dev/ttyS1"   # see WIRING.md's "UART TX/RX" row -- adjust for your specific board
DEFAULT_BAUD = 115200
ACK_TIMEOUT_S = 5
POLL_INTERVAL_S = 1.0


def _open_serial(port, baud):
    import serial  # pyserial -- see requirements_rk3588.txt
    return serial.Serial(port, baudrate=baud, timeout=ACK_TIMEOUT_S)


def _send_line(ser, line, dry_run):
    if dry_run:
        print(f"[dry-run] would send over UART: {line!r}")
        return True
    ser.write((line + "\n").encode("ascii", errors="replace"))
    ack = ser.readline().decode("ascii", errors="replace").strip()
    if ack:
        print(f"[uart_bridge] sent {line!r} -> ACK {ack!r}")
        return ack.upper().startswith("OK") or ack.upper().startswith("ACK")
    print(f"[uart_bridge] sent {line!r} -> no ACK within {ACK_TIMEOUT_S}s (ESP32 unreachable? queued for retry)")
    return False


def run(queue_path, port, baud, dry_run):
    ser = None if dry_run else _open_serial(port, baud)
    print(f"[uart_bridge] watching {queue_path} "
          f"{'(dry-run, no serial port opened)' if dry_run else f'-> {port} @ {baud}'}")
    last_size = 0
    try:
        while True:
            if os.path.isfile(queue_path):
                with open(queue_path, "r", encoding="utf-8") as fh:
                    lines = [ln.strip() for ln in fh.readlines() if ln.strip()]
                if len(lines) > last_size:
                    for line in lines[last_size:]:
                        ok = _send_line(ser, line, dry_run)
                        if not ok:
                            # Leave it in the file (don't advance last_size past it) -- next poll
                            # retries. A GSM outage delays the alert, per ARCHITECTURE.md, not drop it.
                            break
                        last_size += 1
            time.sleep(POLL_INTERVAL_S)
    except KeyboardInterrupt:
        print("\n[uart_bridge] stopped.")
    finally:
        if ser is not None:
            ser.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--queue", default=DEFAULT_QUEUE, help="Alert queue file GUI.py appends to")
    ap.add_argument("--port", default=DEFAULT_PORT, help="UART device, e.g. /dev/ttyS1")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    ap.add_argument("--dry-run", action="store_true", help="Print instead of opening a serial port")
    args = ap.parse_args()

    if not args.dry_run:
        try:
            import serial  # noqa: F401
        except ImportError:
            print("pyserial not installed -- run: pip install -r requirements_rk3588.txt "
                  "(or use --dry-run to test without hardware)", file=sys.stderr)
            sys.exit(1)

    run(args.queue, args.port, args.baud, args.dry_run)


if __name__ == "__main__":
    main()
