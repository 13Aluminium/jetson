#!/usr/bin/env python3
"""
Script 4: Fly + Move (no camera)
==================================
Takeoff 5m → move 1.5m forward → move 1.5m right → LAND
Tests that GUIDED velocity/position commands work.

Terminal 1: mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 --out=udp:127.0.0.1:14551
Terminal 2: python3 4_fly_and_move.py

You can also test this in SITL first:
Terminal 1: sim_vehicle.py -v ArduCopter --console --map --out=udp:127.0.0.1:14551
Terminal 2: python3 4_fly_and_move.py --sitl
"""
import argparse, time
from flight_utils import (FlightController, SafeFlight, TAKEOFF_ALT,
                          confirm, create_log, log)

MOVE_DIST = 1.5    # meters (~5ft)
MOVE_SPEED = 0.5   # m/s
SETTLE = 3.0       # seconds after each move

def move_with_velocity(fc, log_f, vx, vy, vz, duration, label):
    """
    Send body velocity for 'duration' seconds, then stop.
    This is more predictable than position offsets for short moves.
    Distance ≈ speed × time.
    """
    log(log_f, f"Moving: {label} at {(vx**2+vy**2)**0.5:.1f}m/s for {duration:.1f}s")
    t0 = time.time()
    while time.time() - t0 < duration:
        fc.poll()
        fc.velocity_body(vx, vy, vz)
        elapsed = time.time() - t0
        print(f"\r  {label}... {elapsed:.1f}s / {duration:.1f}s | Alt={fc.alt:.1f}m   ",
              end="", flush=True)
        time.sleep(0.2)  # re-send every 200ms (vehicle stops after 3s)
    
    # Stop
    fc.stop()
    print(f"\n[*] {label} done. Settling {SETTLE}s...")
    time.sleep(SETTLE)

def main(args):
    dist = args.distance
    move_time = dist / MOVE_SPEED  # time = distance / speed

    if not args.sitl:
        if not confirm("4_fly_and_move.py",
                       f"Takeoff {args.alt}m → {dist}m FWD → {dist}m RIGHT → LAND"):
            return

    fc = FlightController()
    fc.connect()
    if not args.sitl and not fc.preflight():
        fc.close(); return

    log_fname, log_f = create_log("move")
    log(log_f, f"Alt={args.alt}m | Move={dist}m | Speed={MOVE_SPEED}m/s")

    with SafeFlight(fc) as sf:

        if not fc.set_guided(): return
        if not fc.arm(): return
        log(log_f, "Armed")

        if not fc.takeoff(args.alt): fc.set_rtl(); return
        if not fc.wait_alt(args.alt): fc.set_rtl(); fc.wait_disarmed(); return

        log(log_f, f"At {args.alt}m — stabilizing 3s")
        time.sleep(3)

        # ── Move FORWARD ──
        # velocity_body(forward, right, down)
        move_with_velocity(fc, log_f,
                           vx=MOVE_SPEED, vy=0, vz=0,
                           duration=move_time,
                           label=f"{dist}m FORWARD")

        # ── Move RIGHT ──
        move_with_velocity(fc, log_f,
                           vx=0, vy=MOVE_SPEED, vz=0,
                           duration=move_time,
                           label=f"{dist}m RIGHT")

        # ── LAND ──
        log(log_f, "Landing at current position")
        fc.set_land()
        fc.wait_disarmed()
        log(log_f, "Landed!")

    log_f.close()
    print(f"[*] Log: {log_fname}")
    print("[*] Done!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alt", type=float, default=TAKEOFF_ALT)
    p.add_argument("--distance", type=float, default=MOVE_DIST,
                   help="Move distance in meters (default: 1.5 ≈ 5ft)")
    p.add_argument("--sitl", action="store_true")
    main(p.parse_args())
