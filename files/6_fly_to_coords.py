#!/usr/bin/env python3
"""
fly_to_coords.py — Fly straight to target GPS coordinates, hover, land.

Flow:
    1. Connect → fetch current GPS position
    2. Show current coords, target coords, distance
    3. Wait for confirmation
    4. Takeoff to --alt
    5. Fly straight to target lat/lon at --alt
    6. Hover 5 seconds
    7. Land

Usage:
    # SITL
    python3 fly_to_coords.py --lat 33.78310 --lon -118.10940 --alt 5 --sitl

    # Real flight
    python3 fly_to_coords.py --lat 33.78310 --lon -118.10940 --alt 5

    # Faster cruise speed
    python3 fly_to_coords.py --lat 33.78310 --lon -118.10940 --alt 8 --speed 2.0

Terminal 1: mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 --out=udp:127.0.0.1:14551
Terminal 2: python3 fly_to_coords.py --lat <target_lat> --lon <target_lon> --alt <meters>

Failsafes:
    Ctrl+C → RTL | Exception → RTL
"""

import argparse
import math
import sys
import time

from pymavlink import mavutil

from flight_utils import (
    FlightController, SafeFlight, create_log, log, confirm
)

# ── Defaults ──────────────────────────────────────────────────
DEFAULT_ALT   = 5.0     # meters
DEFAULT_SPEED = 1.0     # m/s cruise speed
ARRIVE_RADIUS = 1.0     # meters — "arrived" when this close
HOVER_TIME    = 5.0     # seconds to hover before landing
POLL_RATE     = 0.5     # seconds between position checks
NAV_TIMEOUT   = 300     # seconds max for navigation (safety)


def haversine(lat1, lon1, lat2, lon2):
    """Distance in meters between two GPS points."""
    R = 6_371_000  # Earth radius in meters
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lat1, lon1, lat2, lon2):
    """Initial bearing in degrees from point 1 to point 2."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(rlat2)
    y = (math.cos(rlat1) * math.sin(rlat2) -
         math.sin(rlat1) * math.cos(rlat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def bearing_to_compass(deg):
    """Convert bearing degrees to compass direction."""
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(deg / 22.5) % 16
    return dirs[idx]


def send_goto(fc, lat, lon, alt):
    """
    Command the FC to fly to a global GPS coordinate in GUIDED mode.
    Uses SET_POSITION_TARGET_GLOBAL_INT which is the proper way to
    send a goto command in ArduCopter GUIDED mode.
    """
    fc.master.mav.set_position_target_global_int_send(
        0,                                              # time_boot_ms
        fc.master.target_system,
        fc.master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0b0000_1111_1111_1000,  # type_mask: position only
        int(lat * 1e7),         # lat_int (degE7)
        int(lon * 1e7),         # lon_int (degE7)
        alt,                    # alt (meters, relative)
        0, 0, 0,                # vx, vy, vz (ignored)
        0, 0, 0,                # afx, afy, afz (ignored)
        0, 0                    # yaw, yaw_rate (ignored)
    )


def build_parser():
    p = argparse.ArgumentParser(description="Fly to target GPS coordinates")
    p.add_argument("--lat",   type=float, required=True,
                   help="Target latitude (decimal degrees)")
    p.add_argument("--lon",   type=float, required=True,
                   help="Target longitude (decimal degrees)")
    p.add_argument("--alt",   type=float, default=DEFAULT_ALT,
                   help=f"Flight altitude in meters (default {DEFAULT_ALT})")
    p.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                   help=f"Cruise speed in m/s (default {DEFAULT_SPEED})")
    p.add_argument("--sitl",  action="store_true", help="SITL mode")
    return p


def main():
    args = build_parser().parse_args()
    target_lat = args.lat
    target_lon = args.lon

    # ── Connect & fetch current position ──────────────────────
    print("[NAV] Connecting to flight controller …")
    fc = FlightController()
    fc.connect()
    fc.poll()

    # Wait for a decent GPS fix
    print("[NAV] Waiting for GPS fix …")
    t0 = time.time()
    while fc.gps_fix < 3:
        fc.poll()
        elapsed = time.time() - t0
        print(f"\r[NAV] GPS fix: {fc.gps_fix}  sats: {fc.satellites}  "
              f"({elapsed:.0f}s)  ", end="", flush=True)
        if elapsed > 60:
            print("\n[!] ERROR: Could not get GPS fix after 60s")
            fc.close()
            sys.exit(1)
        time.sleep(1)
    print()

    cur_lat = fc.lat
    cur_lon = fc.lon

    # ── Calculate distance & bearing ──────────────────────────
    dist = haversine(cur_lat, cur_lon, target_lat, target_lon)
    brng = bearing(cur_lat, cur_lon, target_lat, target_lon)
    compass = bearing_to_compass(brng)

    # ── Show flight plan ──────────────────────────────────────
    print()
    print("=" * 58)
    print("  FLY-TO-COORDINATES — Flight Plan")
    print("=" * 58)
    print()
    print(f"  CURRENT POSITION:")
    print(f"    Lat:  {cur_lat:.8f}")
    print(f"    Lon:  {cur_lon:.8f}")
    print(f"    Sats: {fc.satellites}   Fix: {fc.gps_fix}")
    print(f"    Batt: {fc.battery_pct}%")
    print()
    print(f"  TARGET POSITION:")
    print(f"    Lat:  {target_lat:.8f}")
    print(f"    Lon:  {target_lon:.8f}")
    print()
    print(f"  FLIGHT DETAILS:")
    print(f"    Distance:  {dist:.1f} m")
    print(f"    Bearing:   {brng:.1f}° ({compass})")
    print(f"    Altitude:  {args.alt} m")
    print(f"    Speed:     {args.speed} m/s")
    print(f"    Est. time: {dist / args.speed:.0f}s (cruise only)")
    print()
    print(f"  PLAN: Takeoff → Fly {dist:.1f}m {compass} → Hover 5s → Land")
    print("=" * 58)
    print()

    # ── Safety check ──────────────────────────────────────────
    if dist > 500:
        print(f"[!] WARNING: Distance is {dist:.0f}m — that's over 500m!")
        print("    Double-check your coordinates.")
        print()

    if dist < 1:
        print(f"[!] Target is only {dist:.2f}m away — already there!")
        fc.close()
        return

    # ── Confirmation ──────────────────────────────────────────
    if not args.sitl:
        if not confirm("fly_to_coords",
                       f"Fly {dist:.1f}m {compass} at {args.alt}m alt → Hover 5s → Land"):
            fc.close()
            return
    else:
        resp = input("[CONFIRM] Proceed? (y/n): ").strip().lower()
        if resp != "y":
            print("Aborted.")
            fc.close()
            return

    # ── Preflight ─────────────────────────────────────────────
    if not args.sitl and not fc.preflight():
        fc.close()
        return

    log_fname, log_f = create_log("flyto")
    log(log_f, f"FLY TO COORDS: ({target_lat:.8f}, {target_lon:.8f}) alt={args.alt}m")
    log(log_f, f"From: ({cur_lat:.8f}, {cur_lon:.8f})")
    log(log_f, f"Distance: {dist:.1f}m  Bearing: {brng:.1f}° ({compass})")

    # ── Set cruise speed ──────────────────────────────────────
    fc.master.mav.command_long_send(
        fc.master.target_system,
        fc.master.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
        0,
        0,              # speed type: 0 = airspeed, 1 = groundspeed
        args.speed,     # speed in m/s
        -1,             # throttle (no change)
        0, 0, 0, 0
    )

    with SafeFlight(fc) as sf:

        # ── Phase 1: Takeoff ──────────────────────────────────
        log(log_f, f"TAKEOFF → {args.alt}m")
        print(f"\n[FLIGHT] Setting GUIDED mode …")
        if not fc.set_guided():
            log(log_f, "ERROR: Could not set GUIDED"); return

        print(f"[FLIGHT] Arming …")
        if not fc.arm():
            log(log_f, "ERROR: Arming failed"); return

        print(f"[FLIGHT] Taking off to {args.alt}m …")
        if not fc.takeoff(args.alt):
            log(log_f, "ERROR: Takeoff failed")
            fc.set_rtl(); return

        if not fc.wait_alt(args.alt, tol=1.0, timeout=30):
            log(log_f, "ERROR: Altitude not reached")
            fc.set_rtl(); return

        log(log_f, f"At altitude: {fc.alt:.1f}m — stabilizing 3s")
        print(f"[FLIGHT] At {fc.alt:.1f}m — stabilizing …")
        t0 = time.time()
        while time.time() - t0 < 3:
            fc.poll()
            time.sleep(0.2)

        # ── Phase 2: Fly to target ────────────────────────────
        log(log_f, f"NAVIGATING to ({target_lat:.8f}, {target_lon:.8f})")
        print(f"[FLIGHT] Flying to target …\n")

        nav_t0 = time.time()
        last_cmd = 0

        while True:
            fc.poll()
            now = time.time()

            # Resend goto command every 2 seconds (ArduPilot needs this)
            if now - last_cmd >= 2.0:
                send_goto(fc, target_lat, target_lon, args.alt)
                last_cmd = now

            # Check distance remaining
            remaining = haversine(fc.lat, fc.lon, target_lat, target_lon)

            # Progress
            pct = max(0, (1 - remaining / dist)) * 100 if dist > 0 else 100
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  {bar} {pct:5.1f}%  "
                  f"rem={remaining:.1f}m  "
                  f"alt={fc.alt:.1f}m  "
                  f"sats={fc.satellites}  "
                  f"batt={fc.battery_pct}%  ",
                  end="", flush=True)

            # Arrived?
            if remaining <= ARRIVE_RADIUS:
                print()
                log(log_f, f"ARRIVED — {remaining:.2f}m from target")
                print(f"\n[FLIGHT] ✓ Arrived at target! ({remaining:.2f}m away)")
                break

            # Timeout safety
            if now - nav_t0 > NAV_TIMEOUT:
                print()
                log(log_f, f"NAV TIMEOUT ({NAV_TIMEOUT}s) — landing at current pos")
                print(f"\n[FLIGHT] ⚠ Navigation timeout — landing here")
                break

            time.sleep(POLL_RATE)

        fc.stop()

        # ── Phase 3: Hover ────────────────────────────────────
        log(log_f, f"HOVER at ({fc.lat:.8f}, {fc.lon:.8f}) for {HOVER_TIME}s")
        print(f"[FLIGHT] Hovering for {HOVER_TIME:.0f}s …")
        t0 = time.time()
        while time.time() - t0 < HOVER_TIME:
            fc.poll()
            elapsed = time.time() - t0
            print(f"\r  Hovering: {elapsed:.1f}s / {HOVER_TIME:.0f}s  "
                  f"alt={fc.alt:.1f}m  "
                  f"pos=({fc.lat:.8f}, {fc.lon:.8f})  ",
                  end="", flush=True)
            time.sleep(0.5)
        print()

        # ── Phase 4: Land ─────────────────────────────────────
        log(log_f, "LANDING")
        print(f"[FLIGHT] Landing …")
        fc.set_land()
        fc.wait_disarmed(timeout=30)

        final_dist = haversine(fc.lat, fc.lon, target_lat, target_lon)
        log(log_f, f"LANDED at ({fc.lat:.8f}, {fc.lon:.8f})")
        log(log_f, f"Landing accuracy: {final_dist:.2f}m from target")

        print(f"\n{'='*55}")
        print(f"  ✓ LANDED")
        print(f"  Final position: ({fc.lat:.8f}, {fc.lon:.8f})")
        print(f"  Accuracy:       {final_dist:.2f}m from target")
        print(f"{'='*55}\n")

    log_f.close()
    print(f"[*] Log: {log_fname}")


if __name__ == "__main__":
    main()
