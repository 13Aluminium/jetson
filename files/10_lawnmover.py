#!/usr/bin/env python3
"""
lawnmower.py — Lawnmower pattern flight test
===============================================
Fly a lawnmower (boustrophedon) pattern to test lane-by-lane coverage.

The drone always moves FORWARD in its body frame. Turns are yaw rotations
on its axis, so "forward" changes direction each lane.

Patterns:
    --lanes 1  Out-and-back: 5m fwd → 180° turn → 5m fwd (back to start)
    --lanes 2  Two lanes:    5m fwd → 90° turn → 1m fwd → 90° turn → 5m fwd
    --lanes 3  Three lanes:  ... extends the zigzag one more lane
    --lanes N  N lanes

    --dir ccw  All turns are CCW (lanes shift LEFT from bird's eye)
    --dir cw   All turns are CW  (lanes shift RIGHT from bird's eye)

Bird's eye view (--lanes 3, --dir ccw, drone starts facing NORTH):

         Lane 3 (N)     Lane 1 (N)
           END            START
            ▲               ▲
            │ 5m            │ 5m
            │               │
    (now N) ●───1m (E)────► ●  (now S)
                            │
                     5m     │
                            │
            ● ◄──1m (W)─── ●  (now E)
    (now S) │
            │ 5m
            ▼
        (this would be Lane 4 if --lanes 4)

    Turn tracking (--dir ccw):
        Lane 1: heading N, move 5m forward
        Turn:   90° CCW → heading W
        Shift:  move 1m forward (goes west)
        Turn:   90° CCW → heading S
        Lane 2: move 5m forward (goes south)
        Turn:   90° CCW → heading E
        Shift:  move 1m forward (goes east)
        Turn:   90° CCW → heading N
        Lane 3: move 5m forward (goes north)

Usage:
    # Out-and-back (lanes=1 uses 180° turn)
    python3 lawnmower.py --alt 5 --lanes 1 --dir ccw
    python3 lawnmower.py --alt 5 --lanes 1 --dir cw --turn-angle 180

    # 2-lane test (1st scenario from drawing)
    python3 lawnmower.py --alt 5 --lanes 2 --dir ccw

    # 3-lane test (2nd scenario from drawing)
    python3 lawnmower.py --alt 5 --lanes 3 --dir ccw

    # Custom distances
    python3 lawnmower.py --alt 8 --lanes 4 --lane-length 10 --lane-spacing 2 --dir cw

    # SITL
    python3 lawnmower.py --alt 5 --lanes 2 --dir ccw --sitl

Terminal 1: mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 \\
            --out=udp:127.0.0.1:14551
Terminal 2: python3 lawnmower.py --alt 5 --lanes 3 --dir ccw

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
DEFAULT_LANE_LENGTH  = 5.0    # meters per lane
DEFAULT_LANE_SPACING = 1.0    # meters between lanes
DEFAULT_MOVE_SPEED   = 0.5    # m/s forward movement
DEFAULT_YAW_SPEED    = 15     # °/s rotation speed
SETTLE_TIME          = 3.0    # seconds to stabilize
VEL_RESEND           = 0.2    # seconds between velocity resends
HEADING_TOL          = 3.0    # degrees — yaw "done" tolerance
YAW_TIMEOUT          = 30.0   # max seconds for one yaw


# ── Yaw helpers (same as fly_and_yaw.py) ─────────────────────

def send_yaw(fc, angle_deg, speed_deg_s, direction, relative=True):
    """
    Send CONDITION_YAW command.
    direction: 1 = CW, -1 = CCW
    """
    fc.master.mav.command_long_send(
        fc.master.target_system,
        fc.master.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        abs(angle_deg),
        speed_deg_s,
        direction,
        1 if relative else 0,
        0, 0, 0
    )


def normalize_heading(h):
    """Wrap heading to 0–360."""
    return h % 360


def wait_yaw_complete(fc, log_f, target_heading, timeout=YAW_TIMEOUT):
    """Poll heading until target reached (within tolerance) or timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        fc.poll()
        current = fc.heading

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


# ── Movement helpers ─────────────────────────────────────────

def move_forward(fc, log_f, distance, speed, label="FORWARD"):
    """
    Move forward in body frame using velocity commands.
    Distance ≈ speed × time.
    """
    duration = distance / speed
    log(log_f, f"MOVE: {label} — {distance:.1f}m at {speed:.1f}m/s ({duration:.1f}s)")

    t0 = time.time()
    while time.time() - t0 < duration:
        fc.poll()
        fc.velocity_body(speed, 0, 0)  # vx=forward, vy=0, vz=0
        elapsed = time.time() - t0
        print(f"\r  {label}… {elapsed:.1f}s / {duration:.1f}s | "
              f"Alt={fc.alt:.1f}m | heading={fc.heading:.1f}°   ",
              end="", flush=True)
        time.sleep(VEL_RESEND)

    fc.stop()
    print(f"\n[*] {label} done.")


def do_yaw(fc, log_f, angle, direction, yaw_speed, current_heading, label="TURN"):
    """
    Execute a yaw rotation and wait for completion.
    Returns the new heading.
    """
    dir_label = "CW" if direction == 1 else "CCW"
    log(log_f, f"{label}: {angle:.0f}° {dir_label} at {yaw_speed}°/s")
    print(f"[YAW] {label}: {angle:.0f}° {dir_label}")

    if direction == 1:
        expected = normalize_heading(current_heading + angle)
    else:
        expected = normalize_heading(current_heading - angle)

    send_yaw(fc, angle, yaw_speed, direction, relative=True)

    timeout = max(angle / yaw_speed * 2, YAW_TIMEOUT)
    success = wait_yaw_complete(fc, log_f, expected, timeout=timeout)

    fc.poll()
    actual = fc.heading

    if success:
        log(log_f, f"{label} DONE — heading={actual:.1f}° (target {expected:.1f}°)")
        print(f"[YAW] ✓ {label} done — heading: {actual:.1f}°")
    else:
        log(log_f, f"{label} TIMEOUT — heading={actual:.1f}° (target {expected:.1f}°)")
        print(f"[YAW] ⚠ {label} timed out — heading: {actual:.1f}°")

    return actual


def settle(fc, duration=SETTLE_TIME, label="Settling"):
    """Hold position for a few seconds."""
    print(f"[*] {label} {duration:.0f}s …")
    t0 = time.time()
    while time.time() - t0 < duration:
        fc.poll()
        time.sleep(0.2)


# ── Plan builder ─────────────────────────────────────────────

def build_plan_description(args):
    """Build a human-readable flight plan string."""
    dir_label = "CCW" if args.dir == "ccw" else "CW"
    turn_angle = args.turn_angle

    if args.lanes == 1:
        return (f"Takeoff {args.alt}m → {args.lane_length}m FWD → "
                f"{turn_angle}° {dir_label} → "
                f"{args.lane_length}m FWD (back) → LAND")

    parts = []
    for lane in range(1, args.lanes + 1):
        parts.append(f"{args.lane_length}m FWD")
        if lane < args.lanes:
            parts.append(f"90° {dir_label}")
            parts.append(f"{args.lane_spacing}m FWD")
            parts.append(f"90° {dir_label}")

    return (f"Takeoff {args.alt}m → " + " → ".join(parts) + " → LAND")


def build_visual_map(args):
    """Print an ASCII bird's-eye map of the planned pattern."""
    dir_label = "CCW" if args.dir == "ccw" else "CW"
    ll = args.lane_length
    ls = args.lane_spacing

    print()
    print("=" * 60)
    print("  FLIGHT PLAN — BIRD'S EYE VIEW")
    print("=" * 60)
    print(f"  Lanes: {args.lanes} | Length: {ll}m | Spacing: {ls}m | "
          f"Turns: {dir_label}")
    print(f"  Move speed: {args.move_speed}m/s | Yaw speed: {args.yaw_speed}°/s")
    print()

    if args.lanes == 1:
        turn = args.turn_angle
        print(f"    START")
        print(f"      │")
        print(f"      │  {ll}m forward")
        print(f"      │")
        print(f"      ▼")
        print(f"      ● ── {turn}° {dir_label} turn")
        print(f"      │")
        print(f"      │  {ll}m forward (back)")
        print(f"      │")
        print(f"      ▼")
        print(f"     END (≈ start)")
        print()
        return

    # Multi-lane: build lane positions
    # Each lane is a vertical line, spaced apart horizontally
    # Direction determines if lanes go LEFT or RIGHT
    # CCW → lanes shift left (west if starting north)
    # CW  → lanes shift right (east if starting north)

    lane_labels = []
    for i in range(args.lanes):
        going_out = (i % 2 == 0)  # odd lanes go opposite direction
        if going_out:
            lane_labels.append(f"Lane {i+1} ↑ ({ll}m)")
        else:
            lane_labels.append(f"Lane {i+1} ↓ ({ll}m)")

    # Determine shift direction label
    if args.dir == "ccw":
        shift_arrow = "◄"
        shift_label = "left"
    else:
        shift_arrow = "►"
        shift_label = "right"

    print(f"    Drone starts facing UP (north). Lanes shift {shift_label}.\n")

    # Simple text representation
    for i in range(args.lanes):
        col = f"    {'  ' * i}"  # indent each lane
        if i % 2 == 0:
            print(f"{col}  ▲ {lane_labels[i]}")
            print(f"{col}  │")
            if i == 0:
                print(f"{col}START")
            else:
                print(f"{col}  │")
        else:
            if i == args.lanes - 1:
                print(f"{col} END")
            print(f"{col}  │")
            print(f"{col}  ▼ {lane_labels[i]}")

        # Show shift between lanes
        if i < args.lanes - 1:
            shift_col = f"    {'  ' * i}"
            print(f"{shift_col}  └── 90° {dir_label} → {ls}m {shift_label} → "
                  f"90° {dir_label} ──┐")
            print()

    print()
    est_time = (args.lanes * ll + (args.lanes - 1) * ls) / args.move_speed
    est_yaw_time = (args.lanes - 1) * 2 * (90 / args.yaw_speed)
    total_est = est_time + est_yaw_time + args.lanes * SETTLE_TIME
    print(f"  Est. flight time: ~{total_est:.0f}s "
          f"(move: {est_time:.0f}s + turns: {est_yaw_time:.0f}s + settle)")
    print(f"  Total distance:   "
          f"{args.lanes * ll + (args.lanes - 1) * ls:.1f}m")
    print("=" * 60)
    print()


# ── Argument parser ──────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Lawnmower pattern flight test",
        epilog="Examples:\n"
               "  python3 lawnmower.py --alt 5 --lanes 1 --dir ccw\n"
               "  python3 lawnmower.py --alt 5 --lanes 2 --dir ccw\n"
               "  python3 lawnmower.py --alt 5 --lanes 3 --dir cw\n"
               "  python3 lawnmower.py --alt 8 --lanes 4 --lane-length 10 "
               "--lane-spacing 2 --dir ccw\n"
               "  python3 lawnmower.py --alt 5 --lanes 1 --dir cw "
               "--turn-angle 180 --sitl\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--alt", type=float, default=TAKEOFF_ALT,
                   help=f"Takeoff altitude in meters (default {TAKEOFF_ALT})")
    p.add_argument("--lanes", type=int, required=True,
                   help="Number of lanes (1=out-and-back, 2+=lawnmower)")
    p.add_argument("--dir", choices=["cw", "ccw"], required=True,
                   help="Turn direction: cw=clockwise, ccw=counter-clockwise")
    p.add_argument("--lane-length", type=float, default=DEFAULT_LANE_LENGTH,
                   help=f"Length of each lane in meters (default {DEFAULT_LANE_LENGTH})")
    p.add_argument("--lane-spacing", type=float, default=DEFAULT_LANE_SPACING,
                   help=f"Spacing between lanes in meters (default {DEFAULT_LANE_SPACING})")
    p.add_argument("--move-speed", type=float, default=DEFAULT_MOVE_SPEED,
                   help=f"Forward speed in m/s (default {DEFAULT_MOVE_SPEED})")
    p.add_argument("--yaw-speed", type=float, default=DEFAULT_YAW_SPEED,
                   help=f"Yaw rotation speed in °/s (default {DEFAULT_YAW_SPEED})")
    p.add_argument("--turn-angle", type=float, default=None,
                   help="Turn angle for --lanes 1 only (default 180°). "
                        "Ignored for multi-lane (always 90°).")
    p.add_argument("--sitl", action="store_true", help="SITL mode")
    return p


# ── Main ─────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()

    if args.lanes < 1:
        print("[!] --lanes must be >= 1")
        return

    # Default turn angle: 180° for single lane, 90° for multi-lane
    if args.turn_angle is None:
        args.turn_angle = 180.0 if args.lanes == 1 else 90.0

    dir_value = -1 if args.dir == "ccw" else 1
    dir_label = "CCW" if args.dir == "ccw" else "CW"

    # ── Show plan ─────────────────────────────────────────────
    build_visual_map(args)
    plan_desc = build_plan_description(args)

    if not args.sitl:
        if not confirm("lawnmower.py", plan_desc):
            return
    else:
        print(f"[PLAN] {plan_desc}\n")
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

    log_fname, log_f = create_log("lawnmower")
    log(log_f, f"LAWNMOWER | Lanes={args.lanes} | Dir={dir_label} | "
               f"Alt={args.alt}m")
    log(log_f, f"Lane length={args.lane_length}m | Spacing={args.lane_spacing}m | "
               f"Move speed={args.move_speed}m/s | Yaw speed={args.yaw_speed}°/s")
    log(log_f, f"Plan: {plan_desc}")

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

        settle(fc, SETTLE_TIME, "Post-takeoff stabilize")

        fc.poll()
        current_heading = fc.heading
        log(log_f, f"Initial heading: {current_heading:.1f}°")
        print(f"[FLIGHT] Initial heading: {current_heading:.1f}°\n")

        # ══════════════════════════════════════════════════════
        #  LANES 1: Simple out-and-back
        # ══════════════════════════════════════════════════════
        if args.lanes == 1:
            # Forward
            move_forward(fc, log_f, args.lane_length, args.move_speed,
                         label=f"Lane 1: {args.lane_length}m FWD")
            settle(fc)

            # Turn
            current_heading = do_yaw(fc, log_f, args.turn_angle, dir_value,
                                     args.yaw_speed, current_heading,
                                     label=f"{args.turn_angle:.0f}° {dir_label}")
            settle(fc)

            # Back
            move_forward(fc, log_f, args.lane_length, args.move_speed,
                         label=f"Return: {args.lane_length}m FWD")
            settle(fc)

        # ══════════════════════════════════════════════════════
        #  LANES 2+: Lawnmower pattern
        # ══════════════════════════════════════════════════════
        else:
            for lane in range(1, args.lanes + 1):
                # ── Fly the lane ──
                print(f"\n{'─'*50}")
                log(log_f, f"LANE {lane}/{args.lanes}")
                move_forward(fc, log_f, args.lane_length, args.move_speed,
                             label=f"Lane {lane}/{args.lanes}: "
                                   f"{args.lane_length}m FWD")
                settle(fc)

                # ── Shift to next lane (if not last) ──
                if lane < args.lanes:
                    log(log_f, f"LANE SHIFT {lane}→{lane+1}")
                    print(f"\n[SHIFT] Lane {lane} → {lane + 1}")

                    # First 90° turn
                    current_heading = do_yaw(
                        fc, log_f, 90, dir_value, args.yaw_speed,
                        current_heading,
                        label=f"Shift turn 1: 90° {dir_label}")
                    settle(fc)

                    # Move spacing distance (now perpendicular)
                    move_forward(fc, log_f, args.lane_spacing, args.move_speed,
                                 label=f"Shift: {args.lane_spacing}m FWD")
                    settle(fc)

                    # Second 90° turn (same direction → now facing opposite
                    # of original lane direction)
                    current_heading = do_yaw(
                        fc, log_f, 90, dir_value, args.yaw_speed,
                        current_heading,
                        label=f"Shift turn 2: 90° {dir_label}")
                    settle(fc)

        # ── Summary ───────────────────────────────────────────
        fc.poll()
        final_heading = fc.heading
        net = normalize_heading(final_heading - fc.heading)

        print(f"\n{'='*55}")
        print(f"  LAWNMOWER COMPLETE")
        print(f"  Lanes flown:    {args.lanes}")
        print(f"  Final heading:  {final_heading:.1f}°")
        print(f"  Position:       ({fc.lat:.8f}, {fc.lon:.8f})")
        print(f"  Alt:            {fc.alt:.1f}m")
        print(f"{'='*55}\n")
        log(log_f, f"PATTERN COMPLETE | heading={final_heading:.1f}° "
                    f"pos=({fc.lat:.8f}, {fc.lon:.8f})")

        # ── Land ──────────────────────────────────────────────
        log(log_f, "LANDING at current position")
        print("[FLIGHT] Landing …")
        fc.set_land()
        fc.wait_disarmed(timeout=30)
        log(log_f, f"LANDED at ({fc.lat:.8f}, {fc.lon:.8f})")

    log_f.close()
    print(f"[*] Log: {log_fname}")
    print("[*] Done!")


if __name__ == "__main__":
    main()
