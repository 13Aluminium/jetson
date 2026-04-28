#!/usr/bin/env python3
"""
fly_and_log_gps.py — Takeoff, move forward, log GPS the entire time, land.

Logs GPS coordinates continuously from arm through landing into a CSV file.

Usage:
    # SITL
    python3 fly_and_log_gps.py --sitl --alt 5 --forward 3

    # Real flight
    python3 fly_and_log_gps.py --alt 5 --forward 3

    # Custom speed / log rate
    python3 fly_and_log_gps.py --alt 8 --forward 10 --speed 1.0 --rate 5

Arguments:
    --alt       Takeoff altitude in meters          (default: 5.0)
    --forward   Distance to move forward in meters   (default: 3.0)
    --speed     Forward movement speed in m/s        (default: 0.5)
    --rate      GPS log rate in Hz                   (default: 2)
    --output    Output CSV filename                  (auto-generated if omitted)
    --sitl      Use SITL (no camera)
"""

import argparse
import csv
import os
import sys
import time
import threading
from datetime import datetime

from flight_utils import (
    FlightController, SafeFlight, create_log, log, confirm
)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_ALT       = 5.0    # meters
DEFAULT_FORWARD   = 3.0    # meters
DEFAULT_SPEED     = 0.5    # m/s
DEFAULT_LOG_RATE  = 2      # Hz
SETTLE_TIME       = 3.0    # seconds to hover/settle after each phase
VEL_RESEND        = 0.2    # seconds between velocity command resends


# ── GPS logging thread ───────────────────────────────────────────────────────
class GPSLogger:
    """Runs in a background thread, polls FC and writes GPS rows to CSV."""

    def __init__(self, fc, csv_path, rate_hz):
        self.fc = fc
        self.rate_hz = rate_hz
        self.interval = 1.0 / max(rate_hz, 0.1)
        self._running = False
        self._thread = None
        self.count = 0
        self.csv_path = csv_path

        # Open CSV
        self._csv_file = open(csv_path, "w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow([
            "timestamp",        # ISO-8601
            "epoch_s",          # float seconds since epoch
            "lat",              # decimal degrees
            "lon",              # decimal degrees
            "alt_rel_m",        # relative altitude
            "gps_fix",          # fix type (need ≥3)
            "satellites",       # count
            "heading_deg",      # compass heading
            "battery_pct",      # -1 = unknown
            "armed",            # True/False
            "mode",             # e.g. GUIDED
            "phase",            # current flight phase label
        ])
        self._csv_file.flush()
        self._phase = "INIT"

    def set_phase(self, phase):
        """Update the current flight phase label (shown in CSV)."""
        self._phase = phase

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._csv_file.close()

    def _loop(self):
        while self._running:
            loop_start = time.time()
            try:
                now = datetime.now()
                t = time.time()
                row = [
                    now.isoformat(timespec="milliseconds"),
                    f"{t:.3f}",
                    f"{self.fc.lat:.8f}",
                    f"{self.fc.lon:.8f}",
                    f"{self.fc.alt:.2f}",
                    self.fc.gps_fix,
                    self.fc.satellites,
                    f"{self.fc.heading:.1f}",
                    self.fc.battery_pct,
                    self.fc.armed,
                    self.fc.mode_name,
                    self._phase,
                ]
                self._writer.writerow(row)
                self._csv_file.flush()
                self.count += 1
            except Exception as e:
                print(f"[GPS-LOG] Write error: {e}")

            elapsed = time.time() - loop_start
            sleep_time = self.interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


# ── Argument parser ───────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(description="Takeoff, move forward, log GPS")
    p.add_argument("--alt",     type=float, default=DEFAULT_ALT,
                   help=f"Takeoff altitude in meters (default {DEFAULT_ALT})")
    p.add_argument("--forward", type=float, default=DEFAULT_FORWARD,
                   help=f"Forward distance in meters (default {DEFAULT_FORWARD})")
    p.add_argument("--speed",   type=float, default=DEFAULT_SPEED,
                   help=f"Forward speed in m/s (default {DEFAULT_SPEED})")
    p.add_argument("--rate",    type=float, default=DEFAULT_LOG_RATE,
                   help=f"GPS log rate in Hz (default {DEFAULT_LOG_RATE})")
    p.add_argument("--output",  type=str,   default=None,
                   help="Output CSV path (default: gps_flight_<timestamp>.csv)")
    p.add_argument("--sitl",    action="store_true", help="SITL mode")
    return p


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = build_parser().parse_args()

    # ── Pre-flight confirmation ───────────────────────────────────────────────
    if not confirm("fly_and_log_gps",
                   f"Takeoff to {args.alt}m → move FORWARD {args.forward}m "
                   f"at {args.speed} m/s → LAND. GPS logged throughout."):
        print("Aborted.")
        sys.exit(0)

    # ── CSV path ──────────────────────────────────────────────────────────────
    if args.output:
        csv_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"gps_flight_{ts}.csv"

    # ── Connect ───────────────────────────────────────────────────────────────
    fc = FlightController()
    fc.connect()
    fc.poll()
    fc.preflight()

    # ── Start GPS logger ──────────────────────────────────────────────────────
    gps = GPSLogger(fc, csv_path, args.rate)
    gps.start()
    print(f"[GPS-LOG] Logging at {args.rate} Hz → {csv_path}\n")

    move_duration = args.forward / args.speed  # seconds of forward flight

    with SafeFlight(fc) as sf:
        # ── Phase 1: Takeoff ──────────────────────────────────────────────────
        gps.set_phase("TAKEOFF")
        print(f"[FLIGHT] Setting GUIDED mode …")
        if not fc.set_guided():
            print("[FLIGHT] ERROR: Could not set GUIDED mode")
            gps.stop()
            return

        print(f"[FLIGHT] Arming …")
        if not fc.arm():
            print("[FLIGHT] ERROR: Arming failed")
            gps.stop()
            return

        print(f"[FLIGHT] Taking off to {args.alt}m …")
        if not fc.takeoff(args.alt):
            print("[FLIGHT] ERROR: Takeoff command failed")
            gps.stop()
            return

        fc.wait_alt(args.alt, tol=1.0, timeout=30)
        print(f"[FLIGHT] Altitude reached: {fc.alt:.1f}m")

        # Settle
        gps.set_phase("HOVER_PRE")
        print(f"[FLIGHT] Settling for {SETTLE_TIME}s …")
        t0 = time.time()
        while time.time() - t0 < SETTLE_TIME:
            fc.poll()
            time.sleep(0.2)

        # ── Phase 2: Move forward ─────────────────────────────────────────────
        gps.set_phase("MOVING_FWD")
        print(f"[FLIGHT] Moving FORWARD {args.forward}m at {args.speed} m/s "
              f"({move_duration:.1f}s) …")

        t0 = time.time()
        while time.time() - t0 < move_duration:
            fc.velocity_body(args.speed, 0, 0)   # vx=forward, vy=0, vz=0
            fc.poll()
            time.sleep(VEL_RESEND)

        fc.stop()
        print(f"[FLIGHT] Forward move complete")

        # Settle
        gps.set_phase("HOVER_POST")
        print(f"[FLIGHT] Settling for {SETTLE_TIME}s …")
        t0 = time.time()
        while time.time() - t0 < SETTLE_TIME:
            fc.poll()
            time.sleep(0.2)

        # ── Phase 3: Land ─────────────────────────────────────────────────────
        gps.set_phase("LANDING")
        print(f"[FLIGHT] Landing …")
        fc.set_land()
        fc.wait_disarmed(timeout=60)
        gps.set_phase("DONE")
        print(f"[FLIGHT] Landed and disarmed")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    time.sleep(0.5)   # let logger grab a couple more rows
    gps.stop()
    print(f"\n[GPS-LOG] Done — {gps.count} entries saved to {csv_path}")


if __name__ == "__main__":
    main()
