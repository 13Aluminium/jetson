#!/usr/bin/env python3
"""
Script 1: Fly + Record Video
==============================
Takeoff 5m → hover 15s (recording video) → RTL → save video

Terminal 1 (MAVProxy):
    mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 --out=udp:127.0.0.1:14551
Terminal 2 (this script):
    python3 1_fly_and_record.py
    python3 1_fly_and_record.py --hover-time 30
    python3 1_fly_and_record.py --dry-run         # camera only, no flight
    python3 1_fly_and_record.py --sitl             # SITL test (no camera)
"""
import argparse, time, cv2
from datetime import datetime
from flight_utils import (FlightController, SafeFlight, open_camera,
                          get_camera_fps, TAKEOFF_ALT, confirm)

def main(args):
    if not args.dry_run and not args.sitl:
        if not confirm("1_fly_and_record.py",
                       f"Takeoff {args.alt}m → hover {args.hover_time}s (video) → RTL"):
            return

    # Connect
    fc = FlightController()
    if not args.dry_run:
        fc.connect()
        if not args.sitl and not fc.preflight():
            fc.close(); return

    # Camera
    cap = open_camera(sitl=args.sitl)
    
    # Video writer
    vw = None
    video_fname = f"flight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    if cap:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        # Use measured FPS (not the lying CAP_PROP_FPS)
        rec_fps = args.fps or get_camera_fps(cap, sitl=args.sitl)
        vw = cv2.VideoWriter(video_fname, cv2.VideoWriter_fourcc(*'mp4v'),
                             rec_fps, (w,h))
        print(f"[VID] Recording → {video_fname} ({rec_fps:.1f} FPS)")

    with SafeFlight(fc, camera=cap, video_writer=vw) as sf:

        if args.dry_run:
            print("[DRY] Recording camera only...")
            start = time.time()
            frames = 0
            while time.time() - start < args.hover_time:
                if cap:
                    ret, frame = cap.read()
                    if ret and vw: vw.write(frame); frames += 1
                print(f"\r  {time.time()-start:.1f}s / {args.hover_time}s ({frames} frames)",
                      end="", flush=True)
                time.sleep(0.01)
            print(f"\n[DRY] Done. {frames} frames → {video_fname}")
            return

        # ── FLIGHT ──
        if not fc.set_guided(): return
        if not fc.arm(): return
        if not fc.takeoff(args.alt):
            fc.set_rtl(); return
        if not fc.wait_alt(args.alt):
            fc.set_rtl(); fc.wait_disarmed(); return

        # Hover + record
        print(f"\n[*] Hovering at {args.alt}m — recording {args.hover_time}s...")
        start = time.time()
        frames = 0
        while time.time() - start < args.hover_time:
            fc.poll()
            if cap:
                ret, frame = cap.read()
                if ret and vw: vw.write(frame); frames += 1
            elapsed = time.time() - start
            print(f"\r  {elapsed:.1f}s / {args.hover_time}s | "
                  f"Alt={fc.alt:.1f}m | {frames} frames",
                  end="", flush=True)
            time.sleep(0.01)

        total_time = time.time() - start
        actual_fps = frames / total_time if total_time > 0 else 0
        print(f"\n\n[*] Hover done. {frames} frames in {total_time:.1f}s "
              f"(actual: {actual_fps:.1f} FPS)")

        # RTL
        fc.set_rtl()
        # Keep recording during descent
        while fc.armed:
            fc.poll()
            if cap:
                ret, frame = cap.read()
                if ret and vw: vw.write(frame); frames += 1
            print(f"\r  RTL... Alt={fc.alt:.1f}m | {frames} frames",
                  end="", flush=True)
            time.sleep(0.05)

        print(f"\n\n[*] Landed! {frames} frames → {video_fname}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alt", type=float, default=TAKEOFF_ALT)
    p.add_argument("--hover-time", type=int, default=15)
    p.add_argument("--fps", type=float, default=None,
                   help="Video FPS (default: auto-detect, usually ~21 for IMX477)")
    p.add_argument("--dry-run", action="store_true", help="Camera only, no flight")
    p.add_argument("--sitl", action="store_true", help="SITL mode (webcam or no cam)")
    main(p.parse_args())
