#!/usr/bin/env python3
"""
record_only.py
==============
Just record video from the camera until you stop the script.

Usage:
    python3 record_only.py
    python3 record_only.py --output my_video.mp4
    python3 record_only.py --sitl
Stop:
    Ctrl+C
"""

import argparse
import signal
import sys
import time
from datetime import datetime

import cv2
from flight_utils import open_camera, get_camera_fps


running = True


def handle_stop(signum, frame):
    global running
    running = False


def main(args):
    global running

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    cap = open_camera(sitl=args.sitl)
    if cap is None:
        print("[!] Could not open camera.")
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    fps = get_camera_fps(cap, sitl=args.sitl)

    output = args.output or f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    writer = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        print("[!] Could not open video writer.")
        cap.release()
        return 1

    print(f"[REC] Recording to {output}")
    print(f"[REC] Resolution: {width}x{height} @ {fps:.1f} FPS")
    print("[REC] Press Ctrl+C to stop.")

    frames = 0
    start = time.time()

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                print("\n[!] Frame grab failed.")
                break

            writer.write(frame)
            frames += 1

            elapsed = time.time() - start
            if elapsed > 0:
                rec_fps = frames / elapsed
                print(
                    f"\r[REC] {elapsed:.1f}s | {frames} frames | actual {rec_fps:.1f} FPS",
                    end="",
                    flush=True,
                )

    finally:
        print("\n[*] Stopping...")
        writer.release()
        cap.release()
        print(f"[*] Saved video: {output}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None, help="Output MP4 filename")
    parser.add_argument("--sitl", action="store_true", help="Use webcam/SITL mode")
    sys.exit(main(parser.parse_args()))
