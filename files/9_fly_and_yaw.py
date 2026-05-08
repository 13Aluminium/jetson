#!/usr/bin/env python3
"""
fly_and_yaw.py — Fly + Rotate (Yaw) on axis
==============================================
Takeoff → stabilize 3s → execute yaw rotation(s) → land

Supports multiple rotations in a single flight.

Usage:
    # Single rotation: 90° clockwise
    python3 fly_and_yaw.py --alt 5 --yaw 90 --dir cw

    # Single rotation: 180° counter-clockwise
    python3 fly_and_yaw.py --alt 5 --yaw 180 --dir ccw

    # Multiple rotations: 90° CW, then 180° CCW, then 270° CW
    python3 fly_and_yaw.py --alt 5 --yaw 90 180 270 --dir cw ccw cw

    # All same direction (dir gets reused for all):
    python3 fly_and_yaw.py --alt 5 --yaw 90 90 180 --dir cw

    # Custom speed (default 15°/s — smooth and slow)
    python3 fly_and_yaw.py --alt 5 --yaw 90 --dir cw --speed 30

    # SITL test
    python3 fly_and_yaw.py --alt 5 --yaw 90 --dir cw --sitl

Terminal 1: mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 \\
            --out=udp:127.0.0.1:14551
Terminal 2: python3 fly_and_yaw.py --alt 5 --yaw 90 --dir cw

Failsafes:
    Ctrl+C → RTL | Exception → RTL
"""

import argparse
import time

from pymavlink import mavutil

from flight_utils import (
    FlightController, SafeFlight, TAKEOFF_ALT,
    confirm, create_log, log,
)

# ── Defaults ──────────────────────────────────────────────────
DEFAULT_YAW_SPEED = 15    # °/s — smooth and deliberate
SETTLE_TIME       = 3.0   # seconds to stabilize between phases
HEADING_TOL       = 3.0   # degrees — "rotation complete" tolerance
YAW_TIMEOUT       = 30.0  # max seconds to wait for a single rotation


def send_yaw(fc, angle_deg, speed_deg_s, direction, relative=True):
    """
    Send a CONDITION_YAW command to rotate the drone.

    Args:
        fc:           FlightController instance
        angle_deg:    rotation amount in degrees (always positive)
        speed_deg_s:  rotation speed in °/s
        direction:    1 = clockwise, -1 = counter-clockwise
        relative:     True = relative to current heading,
                      False = absolute compass heading
    """
    fc.master.mav.command_long_send(
        fc.master.target_system,
        fc.master.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,                          # confirmation
        abs(angle_deg),             # param1: target angle (degrees)
        speed_deg_s,                # param2: angular speed (°/s)
        direction,                  # param3: 1=CW, -1=CCW
        1 if relative else 0,       # param4: 1=relative, 0=absolute
        0, 0, 0                     # params 5-7 (unused)
    )


def wait_yaw_complete(fc, log_f, target_heading, timeout=YAW_TIMEOUT):
    """
    Poll heading until it reaches the target (within tolerance) or timeout.
    target_heading should be 0-360.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        fc.poll()
        current = fc.heading  # 0-360 from GLOBAL_POSITION_INT

        # Angular difference (handles wraparound 359→1 etc)
        diff = abs(current - target_heading)
        if diff > 180:
            diff = 360 - diff

        elapsed = time.time() - t0
        print(f"\r  Rotating… heading={current:5.1f}°  "
              f"target={target_heading:5.1f}°  "
              f"remaining≈{diff:4.1f}°  "
              f"({elapsed:.1f}s)   ",
              end="", flush=True)

        if diff <= HEADING_TOL:
            print()
            return True

        time.sleep(0.1)

    print()
    log(log_f, f"YAW TIMEOUT after {timeout}s "
               f"(heading={fc.heading:.1f}° target={target_heading:.1f}°)")
    return False


def normalize_heading(h):
    """Wrap heading to 0-360."""
    return h % 360


def parse_directions(dir_list, count):
    """
    Expand the direction list to match the number of yaw commands.
    If only one direction given, reuse it for all rotations.
    """
    if len(dir_list) >= count:
        return dir_list[:count]
    # Pad with the last specified direction
    return dir_list + [dir_list[-1]] * (count - len(dir_list))


def build_parser():
    p = argparse.ArgumentParser(
        description="Fly to altitude and rotate (yaw) on axis",
        epilog="Examples:\n"
               "  python3 fly_and_yaw.py --alt 5 --yaw 90 --dir cw\n"
               "  python3 fly_and_yaw.py --alt 5 --yaw 90 180 --dir cw ccw\n"
               "  python3 fly_and_yaw.py --alt 5 --yaw 90 90 90 90 --dir cw\n"
               "  python3 fly_and_yaw.py --alt 5 --yaw 360 --dir ccw --speed 10\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--alt", type=float, default=TAKEOFF_ALT,
                   help=f"Takeoff altitude in meters (default {TAKEOFF_ALT})")
    p.add_argument("--yaw", type=float, nargs="+", required=True,
                   help="Rotation angle(s) in degrees (e.g. 90 180 270)")
    p.add_argument("--dir", nargs="+", choices=["cw", "ccw"],
                   required=True,
                   help="Direction(s): cw=clockwise ccw=counter-clockwise. "
                        "If fewer than --yaw entries, last value is reused.")
    p.add_argument("--speed", type=float, default=DEFAULT_YAW_SPEED,
                   help=f"Rotation speed in °/s (default {DEFAULT_YAW_SPEED})")
    p.add_argument("--sitl", action="store_true", help="SITL mode")
    return p


def main():
    args = build_parser().parse_args()

    yaw_angles = args.yaw
    directions = parse_directions(args.dir, len(yaw_angles))
    dir_labels = [("CW" if d == "cw" else "CCW") for d in directions]
    dir_values = [(1 if d == "cw" else -1) for d in directions]

    # ── Build description for confirmation ────────────────────
    rotation_desc = " → ".join(
        f"{a:.0f}° {l}" for a, l in zip(yaw_angles, dir_labels)
    )
    plan = f"Takeoff {args.alt}m → stabilize → {rotation_desc} → LAND"

    if not args.sitl:
        if not confirm("fly_and_yaw.py", plan):
            return
    else:
        print(f"\n[PLAN] {plan}\n")
        resp = input("[CONFIRM] Proceed? (y/n): ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    # ── Connect ───────────────────────────────────────────────
    fc = FlightController()
    fc.connect()
    if not args.sitl and not fc.preflight():
        fc.close()
        return

    log_fname, log_f = create_log("yaw")
    log(log_f, f"FLY AND YAW | Alt={args.alt}m | Speed={args.speed}°/s")
    log(log_f, f"Rotations: {rotation_desc}")

    with SafeFlight(fc) as sf:

        # ── Takeoff ───────────────────────────────────────────
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

        # ── Stabilize ────────────────────────────────────────
        log(log_f, f"At {fc.alt:.1f}m — stabilizing {SETTLE_TIME}s")
        print(f"[FLIGHT] At {fc.alt:.1f}m — stabilizing {SETTLE_TIME:.0f}s …")
        t0 = time.time()
        while time.time() - t0 < SETTLE_TIME:
            fc.poll()
            time.sleep(0.2)

        # Read initial heading
        fc.poll()
        start_heading = fc.heading
        log(log_f, f"Initial heading: {start_heading:.1f}°")
        print(f"[FLIGHT] Initial heading: {start_heading:.1f}°\n")

        # ── Execute rotations ─────────────────────────────────
        current_heading = start_heading

        for i, (angle, direction, dir_label) in enumerate(
            zip(yaw_angles, dir_values, dir_labels), 1
        ):
            total = len(yaw_angles)
            log(log_f, f"ROTATION {i}/{total}: {angle:.0f}° {dir_label} "
                        f"at {args.speed}°/s")
            print(f"[YAW] Rotation {i}/{total}: "
                  f"{angle:.0f}° {dir_label} at {args.speed}°/s")

            # Calculate expected target heading
            if direction == 1:  # CW
                expected = current_heading + angle
            else:               # CCW
                expected = current_heading - angle
            expected = normalize_heading(expected)

            # Send yaw command
            send_yaw(fc, angle, args.speed, direction, relative=True)

            # Wait for it
            success = wait_yaw_complete(fc, log_f, expected,
                                        timeout=max(angle / args.speed * 2, YAW_TIMEOUT))

            fc.poll()
            actual = fc.heading
            current_heading = actual

            if success:
                log(log_f, f"ROTATION {i} DONE — heading={actual:.1f}° "
                            f"(target was {expected:.1f}°)")
                print(f"[YAW] ✓ Rotation {i} complete — heading: {actual:.1f}°")
            else:
                log(log_f, f"ROTATION {i} TIMEOUT — heading={actual:.1f}° "
                            f"(target was {expected:.1f}°)")
                print(f"[YAW] ⚠ Rotation {i} timed out — heading: {actual:.1f}°")

            # Settle between rotations (if more to come)
            if i < total:
                print(f"[YAW] Settling {SETTLE_TIME:.0f}s before next rotation …")
                t0 = time.time()
                while time.time() - t0 < SETTLE_TIME:
                    fc.poll()
                    time.sleep(0.2)

        # ── Post-rotation summary ─────────────────────────────
        fc.poll()
        final_heading = fc.heading
        total_rotation = normalize_heading(final_heading - start_heading)
        # Show the shorter representation
        if total_rotation > 180:
            total_rotation = total_rotation - 360

        print(f"\n{'='*55}")
        print(f"  YAW COMPLETE")
        print(f"  Start heading:  {start_heading:.1f}°")
        print(f"  Final heading:  {final_heading:.1f}°")
        print(f"  Net rotation:   {total_rotation:+.1f}°")
        print(f"{'='*55}\n")
        log(log_f, f"YAW COMPLETE | start={start_heading:.1f}° "
                    f"final={final_heading:.1f}° net={total_rotation:+.1f}°")

        # ── Stabilize before landing ──────────────────────────
        print(f"[FLIGHT] Final stabilize {SETTLE_TIME:.0f}s …")
        t0 = time.time()
        while time.time() - t0 < SETTLE_TIME:
            fc.poll()
            time.sleep(0.2)

        # ── Land ──────────────────────────────────────────────
        log(log_f, "LANDING")
        print("[FLIGHT] Landing …")
        fc.set_land()
        fc.wait_disarmed(timeout=30)
        log(log_f, f"LANDED — heading={fc.heading:.1f}°")

    log_f.close()
    print(f"[*] Log: {log_fname}")
    print("[*] Done!")


if __name__ == "__main__":
    main()
