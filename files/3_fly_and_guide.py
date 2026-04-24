#!/usr/bin/env python3
"""
Script 3: Fly + Guidance (NO movement)
=========================================
Takeoff 5m → hover 2min → detect X → print "move 1.2m RIGHT" → RTL
The drone does NOT move — this verifies the offset→meter math is correct.

Terminal 1: mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 --out=udp:127.0.0.1:14551
Terminal 2: python3 3_fly_and_guide.py
            python3 3_fly_and_guide.py --dry-run
"""
import argparse, time
from flight_utils import (FlightController, SafeFlight, open_camera,
                          load_yolo, detect_x, pixels_to_meters,
                          TAKEOFF_ALT, FRAME_W, FRAME_H,
                          confirm, create_log, log)

def main(args):
    if not args.dry_run:
        if not confirm("3_fly_and_guide.py",
                       f"Takeoff {args.alt}m → show guidance {args.hover_time}s (NO movement) → RTL"):
            return

    model = load_yolo(args.weights, imgsz=args.imgsz)
    fc = FlightController()
    if not args.dry_run:
        fc.connect()
        if not fc.preflight(): fc.close(); return

    cap = open_camera(sitl=args.sitl)
    if not cap: print("[!] No camera."); fc.close(); return

    log_fname, log_f = create_log("guide")
    log(log_f, f"GUIDANCE ONLY — drone will NOT move")
    log(log_f, f"Alt={args.alt}m | Duration={args.hover_time}s | deadzone={args.deadzone}px")

    with SafeFlight(fc, camera=cap) as sf:

        if not args.dry_run:
            if not fc.set_guided(): return
            if not fc.arm(): return
            if not fc.takeoff(args.alt): fc.set_rtl(); return
            if not fc.wait_alt(args.alt): fc.set_rtl(); fc.wait_disarmed(); return
            log(log_f, f"At {args.alt}m — showing guidance")
        else:
            log(log_f, "DRY RUN")

        print(f"\n[*] Guidance for {args.hover_time}s (drone stays still)")
        print(f"    Deadzone: {args.deadzone}px")
        print("-"*65)

        t0 = time.time()
        fps_c = 0; fps_t = time.time(); fps = 0.0; fc_n = 0

        while time.time() - t0 < args.hover_time:
            if not args.dry_run: fc.poll()
            ret, frame = cap.read()
            if not ret: time.sleep(0.05); continue
            fc_n += 1

            cur_alt = fc.alt if (not args.dry_run and fc.alt > 1) else args.alt
            det = detect_x(frame, model, args.conf, args.imgsz)
            elapsed = time.time() - t0
            remain = args.hover_time - elapsed

            if det:
                dx_px = det['cx'] - FRAME_W // 2
                dy_px = det['cy'] - FRAME_H // 2

                if abs(dx_px) <= args.deadzone and abs(dy_px) <= args.deadzone:
                    print(f"\r  [{elapsed:5.1f}s] ★ CENTERED! "
                          f"({dx_px:+d},{dy_px:+d})px | "
                          f"Alt={cur_alt:.1f}m | {remain:.0f}s   ",
                          end="", flush=True)
                    if fc_n % 30 == 0:
                        log(log_f, f"CENTERED ({dx_px:+d},{dy_px:+d})px alt={cur_alt:.1f}m")
                else:
                    m_fwd, m_right = pixels_to_meters(dx_px, dy_px, cur_alt)
                    parts = []
                    if abs(m_fwd) > 0.1:
                        parts.append(f"{abs(m_fwd):.1f}m {'FWD' if m_fwd>0 else 'BACK'}")
                    if abs(m_right) > 0.1:
                        parts.append(f"{abs(m_right):.1f}m {'RIGHT' if m_right>0 else 'LEFT'}")
                    direction = " + ".join(parts) or "~centered"
                    dist = (m_fwd**2 + m_right**2)**0.5

                    print(f"\r  [{elapsed:5.1f}s] → MOVE {direction} "
                          f"({dist:.2f}m) | px=({dx_px:+d},{dy_px:+d}) | "
                          f"Alt={cur_alt:.1f}m | {remain:.0f}s   ",
                          end="", flush=True)
                    if fc_n % 15 == 0:
                        log(log_f, f"MOVE {direction} fwd={m_fwd:+.2f}m right={m_right:+.2f}m "
                                   f"px=({dx_px:+d},{dy_px:+d}) alt={cur_alt:.1f}m")
            else:
                print(f"\r  [{elapsed:5.1f}s] ✗ No X | Alt={cur_alt:.1f}m | {remain:.0f}s   ",
                      end="", flush=True)

            fps_c += 1
            if time.time() - fps_t >= 1.0:
                fps = fps_c / (time.time()-fps_t); fps_c = 0; fps_t = time.time()

        print("\n\n[*] Guidance period complete.")
        if not args.dry_run:
            log(log_f, "RTL"); fc.set_rtl(); fc.wait_disarmed(); log(log_f, "Landed")

    log_f.close()
    print(f"[*] Log: {log_fname}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alt", type=float, default=TAKEOFF_ALT)
    p.add_argument("--hover-time", type=int, default=120)
    p.add_argument("--deadzone", type=int, default=50)
    p.add_argument("--weights", default="best_22.pt")
    p.add_argument("--conf", type=float, default=0.50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sitl", action="store_true")
    main(p.parse_args())
