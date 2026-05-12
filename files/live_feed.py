#!/usr/bin/env python3
"""
live_feed.py — Stream Jetson camera to browser via Flask
=========================================================
Run this on your Jetson, then open the URL on your MacBook.

Usage:
    python3 live_feed.py                  # default port 5000
    python3 live_feed.py --port 8080      # custom port
    python3 live_feed.py --sitl           # use webcam (for testing)

Then on your MacBook, open:
    http://<JETSON_IP>:5000

To find your Jetson's IP:
    hostname -I
"""

import argparse
import time
import cv2
from flask import Flask, Response, render_template_string

# ── Camera Setup ──────────────────────────────────────────────

def open_camera_jetson():
    """Open IMX477 via GStreamer (same pipeline as your flight scripts)."""
    pipeline = (
        "nvarguscamerasrc sensor-mode=0 ! "
        "video/x-raw(memory:NVMM), width=3840, height=2160, "
        "framerate=30/1, format=NV12 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=1920, height=1080, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink max-buffers=1 drop=true sync=false"
    )
    print("[CAM] Opening IMX477 (4K → 1080p)...")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[!] GStreamer failed — trying fallback /dev/video0...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[!] No camera found!")
        return None

    # flush initial garbage frames
    for _ in range(10):
        cap.read()

    print("[CAM] Camera ready.")
    return cap


def open_camera_webcam():
    """Fallback for SITL / testing on a Mac or laptop."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] No webcam found!")
        return None
    print("[CAM] Webcam ready.")
    return cap


# ── Flask App ─────────────────────────────────────────────────

app = Flask(__name__)
camera = None  # set in main()

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
            margin: 20px 0 10px;
            font-size: 1.4em;
            color: #0f0;
            font-weight: 500;
        }
        .status {
            font-size: 0.85em;
            color: #888;
            margin-bottom: 15px;
        }
        .status span { color: #0f0; }
        img {
            max-width: 95vw;
            max-height: 80vh;
            border: 2px solid #333;
            border-radius: 6px;
        }
    </style>
</head>
<body>
    <h1>📡 Jetson Live Feed</h1>
    <p class="status">Status: <span>STREAMING</span></p>
    <img src="/video_feed" alt="Live Feed">
</body>
</html>
"""


def generate_frames():
    """Yield MJPEG frames for the browser."""
    while True:
        if camera is None:
            time.sleep(0.1)
            continue

        ret, frame = camera.read()
        if not ret:
            time.sleep(0.01)
            continue

        # encode as JPEG (adjust quality: 50-95, lower = faster streaming)
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ── Main ──────────────────────────────────────────────────────

def main():
    global camera

    parser = argparse.ArgumentParser(description="Stream Jetson camera to browser")
    parser.add_argument("--port", type=int, default=5000, help="Port (default 5000)")
    parser.add_argument("--sitl", action="store_true", help="Use webcam instead of IMX477")
    parser.add_argument("--quality", type=int, default=70,
                        help="JPEG quality 1-100 (lower = faster, default 70)")
    args = parser.parse_args()

    # Open camera
    if args.sitl:
        camera = open_camera_webcam()
    else:
        camera = open_camera_jetson()

    if camera is None:
        print("[!] Cannot start without a camera. Exiting.")
        return

    # Get Jetson IP for convenience
    import subprocess
    try:
        ip = subprocess.check_output("hostname -I", shell=True).decode().strip().split()[0]
    except Exception:
        ip = "localhost"

    print(f"\n{'='*50}")
    print(f"  LIVE FEED RUNNING")
    print(f"  Open on your MacBook:")
    print(f"    http://{ip}:{args.port}")
    print(f"  (or http://localhost:{args.port} on Jetson)")
    print(f"  Ctrl+C to stop")
    print(f"{'='*50}\n")

    try:
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    finally:
        camera.release()
        print("[*] Camera released. Bye!")


if __name__ == "__main__":
    main()
