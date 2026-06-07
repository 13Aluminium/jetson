#!/usr/bin/env python3
"""
waypoint_nav.py — Waypoint Navigation Only
============================================
Simple 3-waypoint navigation mission:

    1. Takeoff to 52 ft (15.85 m)
    2. Stabilize 5 s for EKF alignment
    3. Fly to WP4 → WP1 → WP7 at 5 m/s
    4. RTL after reaching WP7

No camera, no YOLO, no ball drop — just pure waypoint navigation.

Hardcoded waypoints (CUASC 2026):
    WP4: (35.050101, -118.150687)
    WP1: (35.049875, -118.150127)
    WP7: (35.048843, -118.151194)

Usage:
    # Real flight
    python3 waypoint_nav.py

    # SITL
    python3 waypoint_nav.py --sitl

Terminal 1: mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 \\
            --out=udp:127.0.0.1:14551
Terminal 2: python3 waypoint_nav.py

Failsafes:
    Ctrl+C → RTL | Exception → RTL
"""

import argparse
import math
import sys
import time

from pymavlink import mavutil
from flight_utils import FlightController, SafeFlight, create_log, log, confirm


# ===========================================================================
# HARDCODED MISSION PARAMETERS — DO NOT CHANGE
# ===========================================================================

NAV_ALT      = 15.85    # 52 feet in meters (52 × 0.3048)
NAV_SPEED    = 5.0      # m/s cruise speed
ARRIVE_RADIUS = 2.0     # meters — "arrived" when this close
NAV_TIMEOUT  = 120      # seconds max per waypoint leg
STABILIZE_S  = 5.0      # seconds to hover after takeoff for EKF

WAYPOINTS = [
    {"id": 4, "lat": 35.050101, "lon": -118.150687},
    {"id": 1, "lat": 35.049875, "lon": -118.150127},
    {"id": 7, "lat": 35.048843, "lon": -118.151194},
]


# ===========================================================================
# GPS HELPERS
# ===========================================================================

def haversine(lat1, lon1, lat2, lon2):
    """Distance in meters between two GPS points."""
    R = 6_371_000
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
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[round(deg / 22.5) % 16]


def send_goto(fc, lat, lon, alt):
    """Command the FC to fly to a global GPS coordinate in GUIDED mode."""
    fc.master.mav.set_position_target_global_int_send(
        0, fc.master.target_system, fc.master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0b0000_1111_1111_1000,
        int(lat * 1e7), int(lon * 1e7), alt,
        0, 0, 0, 0, 0, 0, 0, 0)


def set_speed(fc, speed_mps):
    """Set cruise speed via DO_CHANGE_SPEED."""
    fc.master.mav.command_long_send(
        fc.master.target_system, fc.master.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 0,
        0, speed_mps, -1, 0, 0, 0, 0)


# ===========================================================================
# MAIN
# ===========================================================================

def main(args):

    # ── Build flight plan display ────────────────────────────
    print()
    print("=" * 60)
    print("  WAYPOINT NAVIGATION — Flight Plan")
    print("=" * 60)
    print()
    print(f"  ALTITUDE:   {NAV_ALT:.2f} m  (52 ft)")
    print(f"  SPEED:      {NAV_SPEED} m/s")
    print(f"  WAYPOINTS:  {len(WAYPOINTS)}")
    print()
    for i, wp in enumerate(WAYPOINTS):
        tag = "START " if i == 0 else ("FINAL " if i == len(WAYPOINTS) - 1 else "      ")
        print(f"    {tag}WP{wp['id']}:  {wp['lat']:.6f}, {wp['lon']:.6f}")
        if i < len(WAYPOINTS) - 1:
            nxt = WAYPOINTS[i + 1]
            d = haversine(wp['lat'], wp['lon'], nxt['lat'], nxt['lon'])
            b = bearing(wp['lat'], wp['lon'], nxt['lat'], nxt['lon'])
            print(f"           ↓  {d:.1f}m {bearing_to_compass(b)}  "
                  f"(~{d / NAV_SPEED:.0f}s)")
    print()
    print(f"  PLAN: Takeoff 52ft → WP4 → WP1 → WP7 → RTL")
    print("=" * 60)
    print()

    # ── Confirmation ─────────────────────────────────────────
    if not args.sitl:
        desc = (
            f"WAYPOINT NAVIGATION\n"
            f"  Altitude: {NAV_ALT:.2f}m (52 ft) | Speed: {NAV_SPEED} m/s\n"
            f"  Route:    WP4 → WP1 → WP7 → RTL"
        )
        if not confirm("waypoint_nav.py", desc):
            return
    else:
        resp = input("[CONFIRM] Proceed with SITL flight? (y/n): ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    # ── Flight controller ────────────────────────────────────
    fc = FlightController()
    fc.connect()
    if not args.sitl and not fc.preflight():
        fc.close()
        return

    # ── Log file ─────────────────────────────────────────────
    log_fname, log_f = create_log("waypoint_nav")

    log(log_f, "=" * 60)
    log(log_f, "  WAYPOINT NAVIGATION MISSION")
    log(log_f, "=" * 60)
    log(log_f, f"Altitude: {NAV_ALT:.2f}m (52 ft)")
    log(log_f, f"Speed:    {NAV_SPEED} m/s")
    for wp in WAYPOINTS:
        log(log_f, f"  WP{wp['id']}: ({wp['lat']:.6f}, {wp['lon']:.6f})")
    log(log_f, "")

    with SafeFlight(fc) as sf:
        mission_t0 = time.time()

        # ══════════════════════════════════════════════════════
        # 1. TAKEOFF
        # ══════════════════════════════════════════════════════
        log(log_f, f"TAKEOFF → {NAV_ALT:.1f}m (52 ft)")

        if not fc.set_guided():
            log(log_f, "ERROR: Could not set GUIDED mode")
            fc.close(); return

        if not fc.arm():
            log(log_f, "ERROR: Could not arm")
            fc.close(); return

        if not fc.takeoff(NAV_ALT):
            log(log_f, "ERROR: Takeoff failed → RTL")
            fc.set_rtl(); fc.wait_disarmed(timeout=60)
            fc.close(); return

        if not fc.wait_alt(NAV_ALT):
            log(log_f, "ERROR: Did not reach altitude → RTL")
            fc.set_rtl(); fc.wait_disarmed(timeout=60)
            fc.close(); return

        # ── Stabilize for EKF ────────────────────────────────
        log(log_f, f"At {fc.alt:.1f}m — stabilizing {STABILIZE_S:.0f}s for EKF")
        t0 = time.time()
        while time.time() - t0 < STABILIZE_S:
            fc.poll()
            time.sleep(0.2)

        fc.poll()
        log(log_f, f"GPS: ({fc.lat:.8f}, {fc.lon:.8f}) sats={fc.satellites} "
                   f"fix={fc.gps_fix} batt={fc.battery_pct}%")

        # ── Set cruise speed ─────────────────────────────────
        set_speed(fc, NAV_SPEED)
        log(log_f, f"Speed set to {NAV_SPEED} m/s")

        # ══════════════════════════════════════════════════════
        # 2. FLY WAYPOINTS: WP4 → WP1 → WP7
        # ══════════════════════════════════════════════════════
        total_nav_dist = 0

        for wp_idx, wp in enumerate(WAYPOINTS):
            wp_lat = wp['lat']
            wp_lon = wp['lon']
            wp_id  = wp['id']

            fc.poll()
            dist = haversine(fc.lat, fc.lon, wp_lat, wp_lon)
            brng = bearing(fc.lat, fc.lon, wp_lat, wp_lon)
            compass = bearing_to_compass(brng)

            log(log_f, "")
            log(log_f, "─" * 55)
            log(log_f, f"WAYPOINT {wp_idx + 1}/{len(WAYPOINTS)} → "
                       f"WP{wp_id} ({wp_lat:.6f}, {wp_lon:.6f})")
            log(log_f, f"  Distance: {dist:.1f}m | Bearing: {brng:.0f}° ({compass}) | "
                       f"ETA: ~{dist / NAV_SPEED:.0f}s")
            log(log_f, "─" * 55)

            print(f"\n{'─'*55}")
            print(f"  [{wp_idx+1}/{len(WAYPOINTS)}] Flying to WP{wp_id} — "
                  f"{dist:.1f}m {compass}")
            print(f"{'─'*55}")

            nav_t0 = time.time()
            last_cmd = 0

            while True:
                fc.poll()
                now = time.time()

                # Resend goto every 2 seconds
                if now - last_cmd >= 2.0:
                    send_goto(fc, wp_lat, wp_lon, NAV_ALT)
                    last_cmd = now

                # Check distance remaining
                remaining = haversine(fc.lat, fc.lon, wp_lat, wp_lon)
                pct = max(0, (1 - remaining / dist)) * 100 if dist > 0 else 100
                elapsed = now - mission_t0

                # Progress bar
                bar_len = 30
                filled = int(bar_len * pct / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  {bar} {pct:5.1f}%  "
                      f"rem={remaining:.1f}m  "
                      f"alt={fc.alt:.1f}m  "
                      f"sats={fc.satellites}  "
                      f"batt={fc.battery_pct}%  "
                      f"T+{elapsed:.0f}s  ",
                      end="", flush=True)

                # Arrived?
                if remaining <= ARRIVE_RADIUS:
                    print()
                    log(log_f, f"✓ ARRIVED at WP{wp_id} — {remaining:.2f}m from target")
                    total_nav_dist += dist
                    break

                # Timeout safety
                if now - nav_t0 > NAV_TIMEOUT:
                    print()
                    log(log_f, f"⚠ WP{wp_id} TIMEOUT ({NAV_TIMEOUT}s) — "
                               f"moving on ({remaining:.1f}m remaining)")
                    total_nav_dist += (dist - remaining)
                    break

                time.sleep(0.5)

            # Brief hover at waypoint (1s stabilize)
            fc.stop()
            time.sleep(1.0)

        # ══════════════════════════════════════════════════════
        # 3. RTL
        # ══════════════════════════════════════════════════════
        log(log_f, "")
        log(log_f, "=" * 55)
        log(log_f, "  ALL WAYPOINTS REACHED — commanding RTL")
        log(log_f, "=" * 55)

        print(f"\n{'='*55}")
        print(f"  ✓ All waypoints reached — RTL")
        print(f"{'='*55}\n")

        fc.set_rtl()

        # Wait for landing (disarm)
        rtl_t0 = time.time()
        while fc.armed and (time.time() - rtl_t0 < 120):
            fc.poll()
            elapsed = time.time() - mission_t0
            print(f"\r  [RTL] alt={fc.alt:.1f}m  "
                  f"sats={fc.satellites}  "
                  f"batt={fc.battery_pct}%  "
                  f"T+{elapsed:.0f}s   ",
                  end="", flush=True)
            time.sleep(0.5)
        print()

        # ══════════════════════════════════════════════════════
        # MISSION COMPLETE
        # ══════════════════════════════════════════════════════
        mission_total = time.time() - mission_t0

        log(log_f, "")
        log(log_f, "=" * 60)
        log(log_f, "  ★ ★ ★  WAYPOINT NAV COMPLETE  ★ ★ ★")
        log(log_f, "=" * 60)
        log(log_f, f"  Route:        WP4 → WP1 → WP7")
        log(log_f, f"  Altitude:     {NAV_ALT:.1f}m (52 ft)")
        log(log_f, f"  Speed:        {NAV_SPEED} m/s")
        log(log_f, f"  Nav distance: {total_nav_dist:.1f}m")
        log(log_f, f"  Mission time: {mission_total:.1f}s")
        log(log_f, f"  Battery:      {fc.battery_pct}%")
        log(log_f, "=" * 60)

        print(f"\n{'='*60}")
        print(f"  ★ ★ ★  MISSION COMPLETE  ★ ★ ★")
        print(f"  Route:        WP4 → WP1 → WP7 ✓")
        print(f"  Nav distance: {total_nav_dist:.1f}m")
        print(f"  Mission time: {mission_total:.1f}s")
        print(f"  Battery:      {fc.battery_pct}%")
        print(f"{'='*60}\n")

    # ── Close ─────────────────────────────────────────────────
    log_f.close()
    print(f"[*] Flight log: {log_fname}")
    print(f"[*] Done!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Waypoint Navigation — Takeoff 52ft → WP4 → WP1 → WP7 → RTL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Hardcoded mission:
  Altitude: 52 ft (15.85 m) | Speed: 5 m/s
  WP4: (35.050101, -118.150687)
  WP1: (35.049875, -118.150127)
  WP7: (35.048843, -118.151194)

Examples:
  python3 waypoint_nav.py            # Real flight
  python3 waypoint_nav.py --sitl     # SITL testing
""")
    p.add_argument("--sitl", action="store_true",
                   help="SITL mode (simulated FC)")
    main(p.parse_args())
