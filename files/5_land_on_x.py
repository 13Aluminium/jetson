#!/usr/bin/env python3
"""
Script 5: LAND ON X — The Mother Script
==========================================
Takeoff → Search for X → Center over X → Descend in steps → Land on X

State machine:
    TAKEOFF → SEARCH → ACQUIRE → DESCEND → FINAL → LAND

Terminal 1: mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 --out=udp:127.0.0.1:14551
Terminal 2: python3 5_land_on_x.py

SITL test (no real camera — uses webcam or skips detection):
    Terminal 1: sim_vehicle.py -v ArduCopter --console --map --out=udp:127.0.0.1:14551
    Terminal 2: python3 5_land_on_x.py --sitl

Failsafes:
    Ctrl+C → RTL | Exception → RTL | X lost 10s → RTL | Search timeout 60s → RTL
"""
import argparse, time, math
from flight_utils import (FlightController, SafeFlight, open_camera,
                          load_yolo, detect_x, pixels_to_meters,
                          TAKEOFF_ALT, FRAME_W, FRAME_H,
                          confirm, create_log, log)

# ── Tunable parameters ────────────────────────────────────────
DESCEND_STEP = 1.0       # meters per step
FINAL_ALT = 1.5          # switch to final approach below this
LAND_ALT = 0.8           # trigger LAND below this

DEADZONE_HIGH = 60       # px — centered at high altitude
DEADZONE_LOW = 30        # px — centered near ground
SPEED_HIGH = 0.3         # m/s — centering speed at altitude
SPEED_LOW = 0.15         # m/s — centering speed near ground

LOST_TIMEOUT = 10.0      # seconds lost → RTL
SEARCH_TIMEOUT = 60.0    # seconds searching → RTL
DESCENT_VZ = 0.3         # m/s descent rate
VEL_RATE = 0.2           # seconds between velocity commands

def main(args):
    if not args.dry_run and not args.sitl:
        if not confirm("5_land_on_x.py — MOTHER SCRIPT",
                       f"Takeoff {args.alt}m → Find X → Center → Descend → LAND ON X"):
            return

    model = load_yolo(args.weights)

    fc = FlightController()
    if not args.dry_run:
        fc.connect()
        if not args.sitl and not fc.preflight(): fc.close(); return

    cap = open_camera(sitl=args.sitl)
    if not cap and not args.sitl:
        print("[!] No camera — cannot detect X."); fc.close(); return

    log_fname, log_f = create_log("landing")
    log(log_f, "AUTONOMOUS X LANDING")
    log(log_f, f"Alt={args.alt}m | Step={DESCEND_STEP}m | Final={FINAL_ALT}m | Land={LAND_ALT}m")

    with SafeFlight(fc, camera=cap) as sf:

        state = "TAKEOFF"
        last_x = 0          # last time X was seen
        search_t0 = 0       # when search started
        descend_tgt = 0     # target alt for current descent

        if args.dry_run:
            state = "SEARCH"
            log(log_f, "DRY RUN — skipping takeoff")

        # ══════════════════════════════════════════════════════
        # STATE MACHINE
        # ══════════════════════════════════════════════════════
        while state not in ("DONE", "ABORT"):
            if not args.dry_run: fc.poll()
            cur_alt = fc.alt if (not args.dry_run and fc.alt > 0.3) else args.alt

            # Read camera + detect
            det = None
            if cap:
                ret, frame = cap.read()
                if ret:
                    det = detect_x(frame, model, args.conf, args.imgsz)

            # ── TAKEOFF ──────────────────────────────────────
            if state == "TAKEOFF":
                log(log_f, f"TAKEOFF → {args.alt}m")
                if not fc.set_guided(): state = "ABORT"; continue
                if not fc.arm(): state = "ABORT"; continue
                if not fc.takeoff(args.alt):
                    fc.set_rtl(); state = "ABORT"; continue
                if not fc.wait_alt(args.alt):
                    fc.set_rtl(); state = "ABORT"; continue
                log(log_f, "At altitude — stabilizing 3s")
                time.sleep(3)
                state = "SEARCH"
                search_t0 = time.time()
                continue

            # ── SEARCH ───────────────────────────────────────
            elif state == "SEARCH":
                if search_t0 == 0: search_t0 = time.time()
                elapsed = time.time() - search_t0

                if det:
                    log(log_f, f"X FOUND conf={det['conf']:.2f} @({det['cx']},{det['cy']})")
                    state = "ACQUIRE"
                    last_x = time.time()
                    continue

                if elapsed > SEARCH_TIMEOUT:
                    log(log_f, f"SEARCH TIMEOUT ({SEARCH_TIMEOUT}s) → RTL")
                    if not args.dry_run: fc.set_rtl()
                    state = "ABORT"; continue

                print(f"\r  [SEARCH] {elapsed:.0f}s / {SEARCH_TIMEOUT:.0f}s | "
                      f"Alt={cur_alt:.1f}m   ", end="", flush=True)
                time.sleep(0.05)

            # ── ACQUIRE (center on X) ────────────────────────
            elif state == "ACQUIRE":
                if det is None:
                    lost = time.time() - last_x
                    if lost > LOST_TIMEOUT:
                        log(log_f, f"LOST X {lost:.0f}s → RTL")
                        if not args.dry_run: fc.stop(); fc.set_rtl()
                        state = "ABORT"; continue
                    if not args.dry_run: fc.stop()
                    print(f"\r  [ACQUIRE] Lost X — holding ({lost:.1f}s / "
                          f"{LOST_TIMEOUT:.0f}s)   ", end="", flush=True)
                    time.sleep(VEL_RATE)
                    continue

                last_x = time.time()
                dx_px = det['cx'] - FRAME_W // 2
                dy_px = det['cy'] - FRAME_H // 2

                # Adaptive deadzone + speed based on altitude
                dz = DEADZONE_LOW if cur_alt < FINAL_ALT + 1 else DEADZONE_HIGH
                spd = SPEED_LOW if cur_alt < FINAL_ALT + 1 else SPEED_HIGH

                if abs(dx_px) <= dz and abs(dy_px) <= dz:
                    # CENTERED
                    log(log_f, f"CENTERED at {cur_alt:.1f}m "
                               f"(offset: {dx_px:+d},{dy_px:+d}px)")
                    if not args.dry_run: fc.stop()

                    if cur_alt <= LAND_ALT + 0.5:
                        state = "LAND"
                    elif cur_alt <= FINAL_ALT + 0.5:
                        state = "FINAL"
                    else:
                        state = "DESCEND"
                        descend_tgt = max(cur_alt - DESCEND_STEP, FINAL_ALT)
                    time.sleep(0.5)
                    continue

                # Compute correction
                m_fwd, m_right = pixels_to_meters(dx_px, dy_px, cur_alt)
                dist = math.sqrt(m_fwd**2 + m_right**2)
                scale = min(spd / dist, 1.0) if dist > spd else 0.5
                vx = m_fwd * scale
                vy = m_right * scale

                if not args.dry_run:
                    fc.velocity_body(vx, vy, 0)

                parts = []
                if abs(m_fwd)>0.05: parts.append(f"{'FWD' if m_fwd>0 else 'BACK'} {abs(m_fwd):.2f}m")
                if abs(m_right)>0.05: parts.append(f"{'RIGHT' if m_right>0 else 'LEFT'} {abs(m_right):.2f}m")
                print(f"\r  [ACQUIRE] {' + '.join(parts) or '~'} | "
                      f"v=({vx:.2f},{vy:.2f}) | Alt={cur_alt:.1f}m | "
                      f"conf={det['conf']:.2f}   ", end="", flush=True)
                time.sleep(VEL_RATE)

            # ── DESCEND ──────────────────────────────────────
            elif state == "DESCEND":
                log(log_f, f"DESCEND {cur_alt:.1f}m → {descend_tgt:.1f}m")
                t0 = time.time()
                while True:
                    if not args.dry_run: fc.poll()
                    cur_alt = fc.alt if not args.dry_run else descend_tgt
                    if cur_alt <= descend_tgt + 0.3: break
                    if time.time() - t0 > 15: break
                    if not args.dry_run: fc.velocity_ned(0, 0, DESCENT_VZ)
                    print(f"\r  [DESCEND] {cur_alt:.1f}m → {descend_tgt:.1f}m   ",
                          end="", flush=True)
                    time.sleep(VEL_RATE)
                if not args.dry_run: fc.stop()
                log(log_f, f"At {cur_alt:.1f}m — re-acquiring")
                time.sleep(1)
                state = "ACQUIRE"

            # ── FINAL APPROACH ───────────────────────────────
            elif state == "FINAL":
                log(log_f, f"FINAL APPROACH at {cur_alt:.1f}m")
                t0 = time.time()
                while True:
                    if not args.dry_run: fc.poll()
                    cur_alt = fc.alt if not args.dry_run else LAND_ALT

                    if cap:
                        ret, frame = cap.read()
                        if ret: det = detect_x(frame, model, args.conf, args.imgsz)

                    if cur_alt <= LAND_ALT + 0.3:
                        log(log_f, f"Below {LAND_ALT}m → LAND"); state = "LAND"; break
                    if time.time() - t0 > 30:
                        log(log_f, "Final timeout → LAND"); state = "LAND"; break

                    if det:
                        last_x = time.time()
                        dx_px = det['cx'] - FRAME_W//2
                        dy_px = det['cy'] - FRAME_H//2
                        m_fwd, m_right = pixels_to_meters(dx_px, dy_px, cur_alt)
                        dist = math.sqrt(m_fwd**2 + m_right**2)
                        sc = min(SPEED_LOW/dist, 1.0) if dist > SPEED_LOW else 0.4
                        vx, vy = m_fwd*sc, m_right*sc
                        if not args.dry_run: fc.velocity_body(vx, vy, 0.15)
                        print(f"\r  [FINAL] Alt={cur_alt:.1f}m | "
                              f"({dx_px:+d},{dy_px:+d})px | "
                              f"v=({vx:.2f},{vy:.2f},0.15)   ",
                              end="", flush=True)
                    else:
                        lost = time.time() - last_x
                        if not args.dry_run: fc.stop()
                        print(f"\r  [FINAL] Lost X ({lost:.1f}s) | "
                              f"Alt={cur_alt:.1f}m   ", end="", flush=True)
                        if lost > 5:
                            log(log_f, "Lost in final 5s → LAND anyway")
                            state = "LAND"; break
                    time.sleep(VEL_RATE)

            # ── LAND ─────────────────────────────────────────
            elif state == "LAND":
                log(log_f, f"LAND at {cur_alt:.1f}m")
                if not args.dry_run:
                    fc.set_land()
                    fc.wait_disarmed(timeout=30)
                log(log_f, "LANDED ON X!")
                print("\n\n" + "="*55)
                print("  ★ ★ ★  LANDED ON X!  ★ ★ ★")
                print("="*55 + "\n")
                state = "DONE"

            # ── ABORT ────────────────────────────────────────
            elif state == "ABORT":
                log(log_f, "ABORTED")
                if not args.dry_run: fc.wait_disarmed(timeout=60)
                state = "DONE"

    log_f.close()
    print(f"[*] Log: {log_fname}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Autonomous X Landing")
    p.add_argument("--alt", type=float, default=TAKEOFF_ALT)
    p.add_argument("--weights", default="best_22.pt")
    p.add_argument("--conf", type=float, default=0.50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sitl", action="store_true")
    main(p.parse_args())
