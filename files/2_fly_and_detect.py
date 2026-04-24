#!/usr/bin/env python3
"""
Script 2: Fly + Detect X
==========================
Takeoff 5m → hover 3min → YOLO detect X → log to file + terminal → RTL

Terminal 1: mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 --out=udp:127.0.0.1:14551
Terminal 2: python3 2_fly_and_detect.py
            python3 2_fly_and_detect.py --hover-time 60
            python3 2_fly_and_detect.py --dry-run         # ground test
"""
import argparse, time
from flight_utils import (FlightController, SafeFlight, open_camera,
                          load_yolo, detect_x, TAKEOFF_ALT,
                          confirm, create_log, log)

def main(args):
    if not args.dry_run:
        if not confirm("2_fly_and_detect.py",
                       f"Takeoff {args.alt}m → detect X for {args.hover_time}s → RTL"):
            return

    print("[*] Loading YOLO model...")
    model = load_yolo(args.weights)

    # Warmup: run one inference so first real frame isn't slow
    import numpy as np
    print("[*] YOLO warmup inference...")
    _warm_t = time.time()
    _dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = model(_dummy, imgsz=args.imgsz, conf=args.conf, verbose=False)
    print(f"[*] Warmup done in {time.time()-_warm_t:.2f}s "
          f"(device: {next(model.model.parameters()).device})")

    fc = FlightController()
    if not args.dry_run:
        fc.connect()
        if not fc.preflight(): fc.close(); return

    cap = open_camera(sitl=args.sitl)
    if cap is None:
        print("[!] No camera — cannot detect. Aborting.")
        fc.close(); return

    log_fname, log_f = create_log("detect")
    log(log_f, f"Alt={args.alt}m | Duration={args.hover_time}s | conf={args.conf}")

    with SafeFlight(fc, camera=cap) as sf:

        if not args.dry_run:
            if not fc.set_guided(): return
            if not fc.arm(): return
            if not fc.takeoff(args.alt): fc.set_rtl(); return
            if not fc.wait_alt(args.alt): fc.set_rtl(); fc.wait_disarmed(); return
            log(log_f, f"At {args.alt}m — starting detection")
        else:
            log(log_f, "DRY RUN — ground test")

        # Detection loop
        print(f"\n[*] Detecting X for {args.hover_time}s...\n" + "-"*60)
        t0 = time.time()
        frames = 0; found = 0; fps_c = 0; fps_t = time.time(); fps = 0.0

        while time.time() - t0 < args.hover_time:
            if not args.dry_run: fc.poll()

            ret, frame = cap.read()
            if not ret: time.sleep(0.05); continue
            frames += 1

            inf_t0 = time.time()
            det = detect_x(frame, model, args.conf, args.imgsz)
            inf_ms = (time.time() - inf_t0) * 1000
            elapsed = time.time() - t0
            remain = args.hover_time - elapsed

            if det:
                found += 1
                if found % 10 == 1:
                    log(None, "")  # newline
                    log(log_f, f"X conf={det['conf']:.2f} @({det['cx']},{det['cy']}) "
                               f"bbox={det['bbox']} alt={fc.alt:.1f}m inf={inf_ms:.0f}ms")
                print(f"\r  [{elapsed:5.1f}s] ✓ X conf={det['conf']:.2f} "
                      f"@({det['cx']},{det['cy']}) FPS={fps:.1f} "
                      f"inf={inf_ms:.0f}ms {remain:.0f}s left   ",
                      end="", flush=True)
            else:
                print(f"\r  [{elapsed:5.1f}s] ✗ No X | FPS={fps:.1f} | "
                      f"inf={inf_ms:.0f}ms | {remain:.0f}s left   ",
                      end="", flush=True)

            fps_c += 1
            if time.time() - fps_t >= 1.0:
                fps = fps_c / (time.time() - fps_t); fps_c = 0; fps_t = time.time()

        # Summary
        rate = found/frames*100 if frames else 0
        summary = (f"\n{'='*60}\n"
                   f"  DETECTION SUMMARY\n"
                   f"  Frames: {frames} | X found: {found} ({rate:.1f}%) | "
                   f"Avg FPS: {frames/(time.time()-t0):.1f}\n"
                   f"{'='*60}")
        print(summary)
        log(log_f, summary)

        if not args.dry_run:
            log(log_f, "RTL")
            fc.set_rtl(); fc.wait_disarmed()
            log(log_f, "Landed")

    log_f.close()
    print(f"[*] Log: {log_fname}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alt", type=float, default=TAKEOFF_ALT)
    p.add_argument("--hover-time", type=int, default=180)
    p.add_argument("--weights", default="best_22.pt")
    p.add_argument("--conf", type=float, default=0.50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sitl", action="store_true")
    main(p.parse_args())
