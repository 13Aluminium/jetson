#!/usr/bin/env python3
"""
fly_and_capture.py — Fly + Capture Photos
============================================
Takeoff → stabilize → capture photos continuously for N seconds → land

Saves each frame as a JPEG in a timestamped folder.

Usage:
    python3 fly_and_capture.py --alt 5
    python3 fly_and_capture.py --alt 5 --duration 20
    python3 fly_and_capture.py --alt 8 --duration 10 --interval 0.5
    python3 fly_and_capture.py --dry-run              # ground test, no flight
    python3 fly_and_capture.py --alt 5 --sitl

Terminal 1: mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 \\
            --out=udp:127.0.0.1:14551
Terminal 2: python3 fly_and_capture.py --alt 5

Failsafes:
    Ctrl+C → RTL | Exception → RTL
"""

import argparse
import os
import time
from datetime import datetime

import cv2

from flight_utils import (
    FlightController, SafeFlight, open_camera, TAKEOFF_ALT,
    confirm, create_log, log,
)

# ── Defaults ──────────────────────────────────────────────────
DEFAULT_DURATION = 10.0   # seconds of capture
DEFAULT_INTERVAL = 0.0    # seconds between frames (0 = as fast as possible)
SETTLE_TIME      = 3.0    # seconds to stabilize


def main():
    args = build_parser().parse_args()

    plan = (f"Takeoff {args.alt}m → capture photos for {args.duration}s → LAND")
    if args.dry_run:
        plan = f"DRY RUN — capture photos for {args.duration}s (no flight)"

    if not args.dry_run and not args.sitl:
        if not confirm("fly_and_capture.py", plan):
            return

    # ── Camera ────────────────────────────────────────────────
    cap = open_camera(sitl=args.sitl)
    if cap is None:
        print("[!] No camera — cannot capture. Aborting.")
        return

    # ── Output folder ─────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output or f"capture_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[CAP] Saving photos to: {out_dir}/")

    # ── Connect ───────────────────────────────────────────────
    fc = FlightController()
    if not args.dry_run:
        fc.connect()
        if not args.sitl and not fc.preflight():
            fc.close()
            cap.release()
            return

    log_fname, log_f = create_log("capture")
    log(log_f, f"FLY AND CAPTURE | Alt={args.alt}m | Duration={args.duration}s | "
               f"Interval={args.interval}s")

    with SafeFlight(fc, camera=cap) as sf:

        if args.dry_run:
            log(log_f, "DRY RUN — ground capture")
            print(f"\n[DRY] Capturing for {args.duration}s …\n")
        else:
            # ── Takeoff ───────────────────────────────────────
            log(log_f, f"TAKEOFF → {args.alt}m")
            if not fc.set_guided():
                log(log_f, "ERROR: Could not set GUIDED"); return
            if not fc.arm():
                log(log_f, "ERROR: Arming failed"); return
            if not fc.takeoff(args.alt):
                log(log_f, "ERROR: Takeoff failed"); fc.set_rtl(); return
            if not fc.wait_alt(args.alt, tol=1.0, timeout=30):
                log(log_f, "ERROR: Altitude not reached")
                fc.set_rtl(); fc.wait_disarmed(); return

            log(log_f, f"At {fc.alt:.1f}m — stabilizing {SETTLE_TIME}s")
            print(f"[FLIGHT] At {fc.alt:.1f}m — stabilizing {SETTLE_TIME:.0f}s …")
            t0 = time.time()
            while time.time() - t0 < SETTLE_TIME:
                fc.poll()
                time.sleep(0.2)

        # ── Capture loop ──────────────────────────────────────
        log(log_f, f"CAPTURE START — {args.duration}s")
        print(f"[CAP] Capturing …\n")

        t0 = time.time()
        count = 0
        last_capture = 0

        while time.time() - t0 < args.duration:
            if not args.dry_run:
                fc.poll()

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            now = time.time()

            # Respect interval (0 = every frame)
            if args.interval > 0 and (now - last_capture) < args.interval:
                continue

            count += 1
            fname = f"frame_{count:04d}.jpg"
            fpath = os.path.join(out_dir, fname)
            cv2.imwrite(fpath, frame)
            last_capture = now

            elapsed = now - t0
            remain = args.duration - elapsed
            alt_str = f"Alt={fc.alt:.1f}m | " if not args.dry_run else ""
            print(f"\r  [{elapsed:5.1f}s] {alt_str}{count} photos | "
                  f"{remain:.0f}s left   ",
                  end="", flush=True)

        total_time = time.time() - t0
        rate = count / total_time if total_time > 0 else 0

        summary = (f"\n\n{'='*55}\n"
                   f"  CAPTURE COMPLETE\n"
                   f"  Photos: {count}\n"
                   f"  Duration: {total_time:.1f}s\n"
                   f"  Rate: {rate:.1f} photos/s\n"
                   f"  Saved to: {out_dir}/\n"
                   f"{'='*55}")
        print(summary)
        log(log_f, summary)

        # ── Land ──────────────────────────────────────────────
        if not args.dry_run:
            log(log_f, "LANDING")
            print("[FLIGHT] Landing …")
            fc.set_land()
            fc.wait_disarmed(timeout=30)
            log(log_f, "LANDED")

    log_f.close()
    print(f"[*] Log: {log_fname}")
    print(f"[*] {count} photos in {out_dir}/")


def build_parser():
    p = argparse.ArgumentParser(
        description="Fly to altitude and capture photos continuously",
        epilog="Examples:\n"
               "  python3 fly_and_capture.py --alt 5\n"
               "  python3 fly_and_capture.py --alt 5 --duration 20\n"
               "  python3 fly_and_capture.py --alt 5 --interval 0.5\n"
               "  python3 fly_and_capture.py --dry-run\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--alt", type=float, default=TAKEOFF_ALT,
                   help=f"Takeoff altitude in meters (default {TAKEOFF_ALT})")
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                   help=f"Capture duration in seconds (default {DEFAULT_DURATION})")
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                   help="Min seconds between captures (default 0 = every frame)")
    p.add_argument("--output", type=str, default=None,
                   help="Output folder (default: capture_<timestamp>)")
    p.add_argument("--dry-run", action="store_true",
                   help="Ground test — capture without flying")
    p.add_argument("--sitl", action="store_true", help="SITL mode")
    return p


if __name__ == "__main__":
    main()
