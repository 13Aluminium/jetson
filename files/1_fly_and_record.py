#!/usr/bin/env python3
"""
Script 1: Fly + Live Feed
==============================
Takeoff 5m → hover (streaming live to browser) → RTL

No video recording — just a live feed you can watch from your MacBook.

Terminal 1 (MAVProxy):
    mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 --out=udp:127.0.0.1:14551
Terminal 2 (this script):
    python3 1_fly_and_stream.py
    python3 1_fly_and_stream.py --hover-time 30
    python3 1_fly_and_stream.py --dry-run         # camera only, no flight
    python3 1_fly_and_stream.py --sitl

Then open on your MacBook:
    http://<JETSON_IP>:5000
"""
import argparse, time, cv2, threading
from datetime import datetime
from flask import Flask, Response, render_template_string
from flight_utils import (FlightController, SafeFlight, open_camera,
                          TAKEOFF_ALT, FRAME_W, FRAME_H, confirm)

# ── Live Feed globals ─────────────────────────────────────────
FEED_QUALITY = 60
OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
latest_frame = None
frame_lock = threading.Lock()
flask_app = Flask(__name__)

FEED_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Fly + Stream — Live</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #111; color: #eee;
            font-family: -apple-system, system-ui, sans-serif;
            display: flex; flex-direction: column;
            align-items: center; min-height: 100vh;
        }
        h1 { margin: 15px 0 5px; font-size: 1.3em; color: #0f0; font-weight: 500; }
        .info { font-size: 0.8em; color: #666; margin-bottom: 10px; }
        .info span { color: #0f0; }
        img {
            max-width: 95vw; max-height: 84vh;
            border: 2px solid #333; border-radius: 6px;
        }
    </style>
</head>
<body>
    <h1>📡 Fly + Stream — Live Feed</h1>
    <p class="info">Status: <span>STREAMING</span></p>
    <img src="/video_feed" alt="Live Feed">
</body>
</html>
"""


def update_feed(frame):
    """Push a frame to the live feed."""
    global latest_frame
    small = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_NEAREST)
    ret, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, FEED_QUALITY])
    if ret:
        with frame_lock:
            latest_frame = buf.tobytes()


def generate_frames():
    while True:
        with frame_lock:
            fb = latest_frame
        if fb is None:
            time.sleep(0.05)
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + fb + b'\r\n'
        )
        time.sleep(0.03)


@flask_app.route('/')
def index():
    return render_template_string(FEED_HTML)


@flask_app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def start_feed_server(port):
    import subprocess
    try:
        ip = subprocess.check_output("hostname -I", shell=True).decode().strip().split()[0]
    except Exception:
        ip = "localhost"

    print(f"\n{'='*55}")
    print(f"  📡 LIVE FEED on http://{ip}:{port}")
    print(f"  Open this on your MacBook to watch!")
    print(f"{'='*55}\n")

    t = threading.Thread(
        target=lambda: flask_app.run(host='0.0.0.0', port=port, threaded=True,
                                      use_reloader=False),
        daemon=True
    )
    t.start()


# ── HUD overlay ──────────────────────────────────────────────

def draw_hud(frame, state, fc, elapsed=0, hover_time=0):
    """Draw a simple flight HUD on the frame."""
    h, w = frame.shape[:2]
    color = (0, 255, 0)

    # Crosshair
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 1)

    lines = [
        f"STATE: {state}",
        f"ALT: {fc.alt:.1f}m",
        f"GPS: {fc.lat:.6f}, {fc.lon:.6f}",
        f"SATS: {fc.satellites}  FIX: {fc.gps_fix}",
        f"BATT: {fc.battery_pct}%",
        f"HDG: {fc.heading:.0f} deg",
    ]
    if hover_time > 0:
        remain = max(hover_time - elapsed, 0)
        lines.append(f"TIME: {elapsed:.0f}s / {hover_time}s ({remain:.0f}s left)")

    y_off = 25
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (11, y_off + i * 24),
                    OVERLAY_FONT, 0.55, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, y_off + i * 24),
                    OVERLAY_FONT, 0.55, color, 1)

    # Timestamp
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    cv2.putText(frame, ts, (w - 160, h - 12),
                OVERLAY_FONT, 0.5, (255, 255, 255), 1)

    return frame


# ── Main ──────────────────────────────────────────────────────

def main(args):
    if not args.dry_run and not args.sitl:
        if not confirm("1_fly_and_stream.py",
                       f"Takeoff {args.alt}m → hover {args.hover_time}s (live feed) → RTL"):
            return

    # Start live feed
    start_feed_server(args.feed_port)

    # Connect
    fc = FlightController()
    if not args.dry_run:
        fc.connect()
        if not args.sitl and not fc.preflight():
            fc.close(); return

    # Camera
    cap = open_camera(sitl=args.sitl)
    if not cap:
        print("[!] No camera — cannot stream.")
        if not args.sitl: fc.close(); return

    with SafeFlight(fc, camera=cap) as sf:

        # ── DRY RUN ──
        if args.dry_run:
            print("[DRY] Streaming camera only...")
            start = time.time()
            while time.time() - start < args.hover_time:
                ret, frame = cap.read()
                if ret:
                    elapsed = time.time() - start
                    overlay = draw_hud(frame, "DRY RUN", fc, elapsed, args.hover_time)
                    update_feed(overlay)
                print(f"\r  {time.time()-start:.1f}s / {args.hover_time}s",
                      end="", flush=True)
                time.sleep(0.01)
            print(f"\n[DRY] Done.")
            return

        # ── FLIGHT ──
        if not fc.set_guided(): return
        if not fc.arm(): return
        if not fc.takeoff(args.alt):
            fc.set_rtl(); return
        if not fc.wait_alt(args.alt):
            fc.set_rtl(); fc.wait_disarmed(); return

        # Hover + stream
        print(f"\n[*] Hovering at {args.alt}m — streaming {args.hover_time}s...")
        start = time.time()
        while time.time() - start < args.hover_time:
            fc.poll()
            if cap:
                ret, frame = cap.read()
                if ret:
                    elapsed = time.time() - start
                    overlay = draw_hud(frame, "HOVER", fc, elapsed, args.hover_time)
                    update_feed(overlay)
            elapsed = time.time() - start
            print(f"\r  {elapsed:.1f}s / {args.hover_time}s | "
                  f"Alt={fc.alt:.1f}m",
                  end="", flush=True)
            time.sleep(0.01)

        print(f"\n\n[*] Hover done. RTL...")

        # RTL — keep streaming
        fc.set_rtl()
        while fc.armed:
            fc.poll()
            if cap:
                ret, frame = cap.read()
                if ret:
                    overlay = draw_hud(frame, "RTL", fc)
                    update_feed(overlay)
            print(f"\r  RTL... Alt={fc.alt:.1f}m",
                  end="", flush=True)
            time.sleep(0.05)

        print(f"\n\n[*] Landed!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fly + Live Stream (no recording)")
    p.add_argument("--alt", type=float, default=TAKEOFF_ALT)
    p.add_argument("--hover-time", type=int, default=15)
    p.add_argument("--dry-run", action="store_true", help="Camera only, no flight")
    p.add_argument("--sitl", action="store_true", help="SITL mode")
    p.add_argument("--feed-port", type=int, default=5000,
                   help="Port for live feed (default 5000)")
    main(p.parse_args())
