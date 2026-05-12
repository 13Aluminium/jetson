#!/usr/bin/env python3
"""
live_feed_fast.py — Low-latency Jetson camera stream
======================================================
Optimizations over the basic version:
  1. Threaded camera reads (never waits for cv2.read in the main loop)
  2. Smaller stream resolution (960x540 instead of 1920x1080)
  3. Lower JPEG quality by default (50)
  4. Skips frames if the client can't keep up
  5. Uses Waitress (production WSGI) instead of Flask dev server

Usage:
    python3 live_feed_fast.py
    python3 live_feed_fast.py --port 8080
    python3 live_feed_fast.py --quality 40 --width 640 --height 360   # potato mode, minimal lag
    python3 live_feed_fast.py --quality 80 --width 1920 --height 1080 # high quality, more lag
    python3 live_feed_fast.py --sitl  # webcam for testing

Then on MacBook:  http://<JETSON_IP>:5000
"""

import argparse
import time
import threading
import cv2
from flask import Flask, Response, render_template_string

# ── Threaded Camera ───────────────────────────────────────────

class CameraStream:
    """Reads frames in a background thread — always has the latest frame ready."""

    def __init__(self, cap, resize_w=None, resize_h=None):
        self.cap = cap
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and self.resize_w and self.resize_h:
                frame = cv2.resize(frame, (self.resize_w, self.resize_h),
                                   interpolation=cv2.INTER_NEAREST)
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def release(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


# ── Camera Setup ──────────────────────────────────────────────

def open_camera_jetson(width, height):
    """
    Open IMX477 via GStreamer.
    ALWAYS captures at 4K sensor-mode=0 (3840x2160) for FULL FOV.
    Then nvvidconv downscales to stream resolution in hardware (GPU/VIC) — nearly free.
    So you get the wide 4K FOV even when streaming at 720p or 1080p.
    """
    pipeline = (
        f"nvarguscamerasrc sensor-mode=0 ! "
        f"video/x-raw(memory:NVMM), width=3840, height=2160, "
        f"framerate=30/1, format=NV12 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink max-buffers=1 drop=true sync=false"
    )
    print(f"[CAM] Opening IMX477 (4K sensor → stream at {width}x{height})...")
    print(f"[CAM] Full 4K FOV preserved — downscale is hardware accelerated")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[!] GStreamer failed — trying /dev/video0...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return None

    # flush garbage frames
    for _ in range(10):
        cap.read()

    print("[CAM] Camera ready.")
    return cap


def open_camera_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    print("[CAM] Webcam ready.")
    return cap


# ── Flask App ─────────────────────────────────────────────────

app = Flask(__name__)
stream = None       # CameraStream, set in main()
jpeg_quality = 50   # set in main()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Jetson Live Feed</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #111;
            color: #eee;
            font-family: -apple-system, system-ui, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        h1 {
            margin: 20px 0 6px;
            font-size: 1.3em;
            color: #0f0;
            font-weight: 500;
        }
        .info {
            font-size: 0.8em;
            color: #666;
            margin-bottom: 12px;
        }
        img {
            max-width: 95vw;
            max-height: 82vh;
            border: 2px solid #333;
            border-radius: 6px;
        }
    </style>
</head>
<body>
    <h1>📡 Jetson Live Feed</h1>
    <p class="info">4K FOV → STREAM_RES stream | Quality STREAM_Q</p>
    <img src="/video_feed" alt="Live Feed">
</body>
</html>
"""


def generate_frames():
    """Yield MJPEG frames as fast as possible."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    while True:
        if stream is None:
            time.sleep(0.05)
            continue

        ret, frame = stream.read()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        ret, buffer = cv2.imencode('.jpg', frame, encode_params)
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )


@app.route('/')
def index():
    page = HTML_PAGE.replace("STREAM_RES", f"{args.width}x{args.height}")
    page = page.replace("STREAM_Q", str(jpeg_quality))
    return render_template_string(page)


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ── Main ──────────────────────────────────────────────────────

args = None

def main():
    global stream, jpeg_quality, args

    parser = argparse.ArgumentParser(description="Low-latency Jetson camera stream (4K FOV)")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--width", type=int, default=1280,
                        help="Stream width (default 1280). Full 4K FOV is always preserved.")
    parser.add_argument("--height", type=int, default=720,
                        help="Stream height (default 720). Use 1920x1080 for sharper.")
    parser.add_argument("--quality", type=int, default=50,
                        help="JPEG quality 1-100 (default 50, lower = less lag)")
    parser.add_argument("--sitl", action="store_true", help="Use webcam")
    args = parser.parse_args()

    jpeg_quality = args.quality

    # Open camera
    if args.sitl:
        raw_cap = open_camera_webcam()
        resize_w, resize_h = args.width, args.height
    else:
        # Let GStreamer do the resize in hardware (faster than CPU resize)
        raw_cap = open_camera_jetson(args.width, args.height)
        resize_w, resize_h = None, None  # already resized by GStreamer

    if raw_cap is None:
        print("[!] No camera. Exiting.")
        return

    # Wrap in threaded reader
    stream = CameraStream(raw_cap, resize_w, resize_h)

    # Get IP
    import subprocess
    try:
        ip = subprocess.check_output("hostname -I", shell=True).decode().strip().split()[0]
    except Exception:
        ip = "localhost"

    print(f"\n{'='*50}")
    print(f"  LIVE FEED (4K FOV)")
    print(f"  Sensor: 3840x2160 (full FOV)")
    print(f"  Stream: {args.width}x{args.height} @ quality {args.quality}")
    print(f"")
    print(f"  Open on your MacBook:")
    print(f"    http://{ip}:{args.port}")
    print(f"")
    print(f"  Options:")
    print(f"    --width 1920 --height 1080    sharper")
    print(f"    --width 640 --height 360      min lag")
    print(f"    --quality 30                  faster")
    print(f"  Ctrl+C to stop")
    print(f"{'='*50}\n")

    try:
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    finally:
        stream.release()
        print("[*] Done.")


if __name__ == "__main__":
    main()
