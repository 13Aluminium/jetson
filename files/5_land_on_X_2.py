#!/usr/bin/env python3
"""
Script 12: LAND ON X — Rolling Window Latch + LIVE BROWSER FEED
=================================================================
Improvement over Script 11 (consecutive-frame latch):

    Script 11's problem:
        Required 8 CONSECUTIVE centered frames at latch altitude.
        One wind gust or YOLO bbox jitter resets the counter to zero.
        In practice this takes 30-60+ seconds or fails entirely.

    Script 12's fix — Rolling Window:
        Instead of demanding a perfect streak, we keep a sliding window
        of the last WINDOW_SIZE frames (~30 frames ≈ 6 seconds).
        Each frame is tagged: centered (within deadzone) or not.
        GPS is recorded for every centered frame.

        Latch triggers when:
          1. At least LATCH_MIN_CENTERED frames are centered within
             the window (e.g. 6 out of 30).
          2. The median GPS of those centered frames is computed.
          3. GPS spread of centered readings < LATCH_MAX_SPREAD.

        One or two bad frames don't reset anything — they just sit
        in the buffer as misses. The window slides forward naturally.

    Result: latch typically completes in 2-5 seconds instead of 30-60+.
    Same accuracy — median GPS rejects outliers better than mean.

Phases (same commit-descent as Script 11):
    Phase 1 (YOLO, above LATCH_ALT):
        Standard YOLO detection → center over X → descend in steps.

    Phase 2 (ROLLING LATCH, at LATCH_ALT):
        Continue centering. Sliding window collects GPS from centered
        frames. Once enough centered frames accumulate → median → latch.

    Phase 3 (COMMIT DESCENT, below LATCH_ALT):
        Stop all vision. GPS position hold → descend → LAND.

Error budget:
    Centering error at latch:  ~0.1-0.2m (25px deadzone at 3m, median)
    GPS drift during descent:  ~0.3-0.7m (Pixhawk 6C position hold)
    Total:                     ~0.5-0.9m → within ±1m tolerance

State machine:
    TAKEOFF → SEARCH → ACQUIRE → DESCEND → ... →
    LATCH_CENTERING → (auto-latch when window fills) →
    COMMIT_DESCENT → LAND

Usage:
    python3 12_land_on_x_rolling_latch.py --alt 5
    python3 12_land_on_x_rolling_latch.py --alt 5 --feed-port 5000
    python3 12_land_on_x_rolling_latch.py --dry-run
    python3 12_land_on_x_rolling_latch.py --sitl

Then open on your MacBook:
    http://<JETSON_IP>:5000

Terminal 1: mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 \\
            --out=udp:127.0.0.1:14551
Terminal 2: python3 12_land_on_x_rolling_latch.py

Failsafes:
    Ctrl+C → RTL | Exception → RTL | X lost 10s → RTL | Search timeout 60s → RTL
"""

import argparse, time, math, os, cv2, threading, statistics
from collections import deque
from datetime import datetime
from flask import Flask, Response, render_template_string
from pymavlink import mavutil
from flight_utils import (FlightController, SafeFlight, open_camera,
                          load_yolo, detect_x, pixels_to_meters,
                          get_camera_fps,
                          TAKEOFF_ALT, FRAME_W, FRAME_H,
                          confirm, create_log, log)

# ===========================================================================
# TUNABLE PARAMETERS
# ===========================================================================

# ── YOLO centering phase (above LATCH_ALT) ────────────────────
DESCEND_STEP     = 1.0       # meters per descent step
DEADZONE_HIGH    = 50        # px — "centered" threshold at high alt
DEADZONE_LATCH   = 25        # px — tighter threshold near latch alt
SPEED_HIGH       = 0.30      # m/s — centering speed at high alt
SPEED_LOW        = 0.15      # m/s — centering speed near latch

# ── Rolling window latch parameters ───────────────────────────
# LATCH_ALT: altitude at which we latch GPS and stop using vision.
# Must be high enough that the FULL X is still visible to YOLO.
# For a 2.5m X prop with 73° HFOV camera:
#   At 3.0m, ground width ≈ 4.4m → 2.5m X fully visible ✓
#   At 2.5m, ground width ≈ 3.6m → 2.5m X barely fits
#   At 2.0m, ground width ≈ 2.9m → X starts overflowing ✗
LATCH_ALT            = 3.0   # altitude to latch GPS (meters)
LATCH_DEADZONE       = 25    # px — must be this centered to count

# ── Rolling window config ─────────────────────────────────────
# WINDOW_SIZE: how many recent frames to keep in the sliding window.
#   At ~5 FPS effective loop rate, 30 frames ≈ 6 seconds of history.
# LATCH_MIN_CENTERED: minimum centered frames within the window to
#   trigger a latch. 6 out of 30 = 20% hit rate — very forgiving.
#   Even in gusty conditions, the drone is centered >50% of frames.
# LATCH_MAX_SPREAD: reject latch if GPS readings are too scattered.
#   Uses median for the latch position, but spread check catches
#   cases where GPS is genuinely unreliable (low sats, multipath).
# LATCH_TIMEOUT: max seconds in LATCH_CENTERING before giving up.
#   Safety net — if GPS is so bad we can't latch in 30s, something
#   is wrong. Falls back to RTL.
WINDOW_SIZE          = 30    # frames in sliding window
LATCH_MIN_CENTERED   = 6    # min centered frames to trigger latch
LATCH_MAX_SPREAD     = 1.5  # meters — max GPS spread (reject if bad)
LATCH_TIMEOUT        = 30.0 # seconds — max time in latch phase

# ── Commit descent parameters ─────────────────────────────────
COMMIT_VZ        = 0.30      # m/s descent rate during commit
COMMIT_CMD_RATE  = 1.0       # seconds between position commands
LAND_ALT         = 0.6       # switch to LAND mode below this

# ── Common ────────────────────────────────────────────────────
LOST_TIMEOUT     = 10.0      # seconds without YOLO detection → RTL
SEARCH_TIMEOUT   = 60.0      # seconds of searching → RTL
VEL_RATE         = 0.2       # seconds between velocity commands
DESCENT_VZ       = 0.30      # m/s — descent rate in YOLO phase

# ── Video / overlay ───────────────────────────────────────────
OVERLAY_FONT         = cv2.FONT_HERSHEY_SIMPLEX
COLOR_OK             = (0, 255, 0)       # green
COLOR_LOST           = (0, 0, 255)       # red
COLOR_CENTER         = (0, 255, 255)     # yellow
COLOR_LATCH          = (255, 165, 0)     # orange
COLOR_COMMIT         = (255, 0, 255)     # magenta


# ===========================================================================
# LIVE FEED (Flask MJPEG server)
# ===========================================================================

FEED_QUALITY = 60
latest_frame = None
frame_lock = threading.Lock()
flask_app = Flask(__name__)

FEED_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Rolling Latch — Live</title>
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
    <h1>🎯 Rolling Latch — Live Feed</h1>
    <p class="info">Status: <span>STREAMING</span> | Rolling window latch landing</p>
    <img src="/video_feed" alt="Live Feed">
</body>
</html>
"""


def update_feed_frame(overlay_frame):
    global latest_frame
    small = cv2.resize(overlay_frame, (960, 540), interpolation=cv2.INTER_NEAREST)
    ret, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, FEED_QUALITY])
    if ret:
        with frame_lock:
            latest_frame = buf.tobytes()


def generate_frames():
    while True:
        with frame_lock:
            frame_bytes = latest_frame
        if frame_bytes is None:
            time.sleep(0.05)
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
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
    return t


# ===========================================================================
# GPS HELPERS
# ===========================================================================

def haversine(lat1, lon1, lat2, lon2):
    """Distance in meters between two GPS points."""
    R = 6_371_000
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def gps_spread(readings):
    """Max distance between any two GPS readings in a list of (lat, lon)."""
    if len(readings) < 2:
        return 0.0
    max_d = 0.0
    for i in range(len(readings)):
        for j in range(i + 1, len(readings)):
            d = haversine(readings[i][0], readings[i][1],
                          readings[j][0], readings[j][1])
            if d > max_d:
                max_d = d
    return max_d


def median_gps(readings):
    """
    Compute the median GPS position from a list of (lat, lon).
    Uses component-wise median — robust to outliers unlike mean.
    """
    if not readings:
        return 0.0, 0.0
    lats = [r[0] for r in readings]
    lons = [r[1] for r in readings]
    return statistics.median(lats), statistics.median(lons)


def send_position_with_descent(fc, lat, lon, alt):
    """
    Command Pixhawk to hold a GPS position at a given altitude.
    Uses SET_POSITION_TARGET_GLOBAL_INT with position mask.
    """
    fc.master.mav.set_position_target_global_int_send(
        0,
        fc.master.target_system,
        fc.master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0b0000_1111_1111_1000,  # type_mask: use position only
        int(lat * 1e7),
        int(lon * 1e7),
        alt,
        0, 0, 0,
        0, 0, 0,
        0, 0
    )


# ===========================================================================
# ROLLING WINDOW LATCH
# ===========================================================================

class RollingLatchWindow:
    """
    Sliding window that tracks recent frames and their GPS readings.

    Each frame is tagged as centered (True) or not (False).
    Only centered frames have their GPS recorded.

    Latch triggers when:
      - At least min_centered frames are centered in the window
      - GPS spread of centered readings < max_spread

    The latch position is the MEDIAN of centered GPS readings,
    which is robust to outliers (unlike mean).
    """

    def __init__(self, window_size=WINDOW_SIZE,
                 min_centered=LATCH_MIN_CENTERED,
                 max_spread=LATCH_MAX_SPREAD):
        self.window_size = window_size
        self.min_centered = min_centered
        self.max_spread = max_spread

        # Each entry: (timestamp, is_centered, lat, lon, dx_px, dy_px)
        self.buffer = deque(maxlen=window_size)

    def add_frame(self, is_centered, lat, lon, dx_px=0, dy_px=0):
        """Record one frame's result into the sliding window."""
        self.buffer.append((time.time(), is_centered, lat, lon, dx_px, dy_px))

    @property
    def total_frames(self):
        return len(self.buffer)

    @property
    def centered_count(self):
        return sum(1 for entry in self.buffer if entry[1])

    @property
    def centered_readings(self):
        """GPS readings from centered frames only."""
        return [(entry[2], entry[3]) for entry in self.buffer if entry[1]]

    @property
    def hit_rate(self):
        """Fraction of frames that are centered (0.0 to 1.0)."""
        if not self.buffer:
            return 0.0
        return self.centered_count / len(self.buffer)

    def check_latch(self):
        """
        Check if latch conditions are met.

        Returns:
            None if not ready to latch.
            dict with latch info if ready:
                {
                    'lat': float,       # median latitude
                    'lon': float,       # median longitude
                    'spread': float,    # GPS spread in meters
                    'n_centered': int,  # number of centered frames used
                    'n_total': int,     # total frames in window
                    'hit_rate': float,  # fraction centered
                    'readings': list,   # the GPS readings used
                }
        """
        n_centered = self.centered_count
        if n_centered < self.min_centered:
            return None

        readings = self.centered_readings
        spread = gps_spread(readings)
        if spread > self.max_spread:
            return None  # GPS too scattered — keep collecting

        lat, lon = median_gps(readings)

        return {
            'lat': lat,
            'lon': lon,
            'spread': spread,
            'n_centered': n_centered,
            'n_total': len(self.buffer),
            'hit_rate': self.hit_rate,
            'readings': readings,
        }

    def reset(self):
        """Clear the window (e.g. if X is lost for a long time)."""
        self.buffer.clear()

    def summary_str(self):
        """Short string for console/log output."""
        n = self.centered_count
        total = len(self.buffer)
        vis = ""
        for entry in self.buffer:
            vis += "●" if entry[1] else "○"
        return f"{vis} {n}/{total} ({self.hit_rate:.0%})"


# ===========================================================================
# VIDEO OVERLAY
# ===========================================================================

def draw_overlay(frame, state, det, cur_alt, fc,
                 centered=False, latch_info=None, commit_info=None):
    """Draw full HUD overlay on the video frame."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # ── Crosshair ────────────────────────────────────────────
    size = 30
    cv2.line(frame, (cx - size, cy), (cx + size, cy), COLOR_CENTER, 1)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), COLOR_CENTER, 1)

    # ── Deadzone circle (adaptive) ───────────────────────────
    if state == "LATCH_CENTERING":
        dz = LATCH_DEADZONE
        dz_color = COLOR_LATCH
    elif cur_alt < LATCH_ALT + 1.5:
        dz = DEADZONE_LATCH
        dz_color = COLOR_CENTER
    else:
        dz = DEADZONE_HIGH
        dz_color = COLOR_CENTER
    cv2.circle(frame, (cx, cy), dz, dz_color, 1)

    # ── YOLO detection ───────────────────────────────────────
    if det:
        x1, y1, x2, y2 = det['bbox']
        color = COLOR_OK
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        dcx, dcy = int(det['cx']), int(det['cy'])
        cv2.line(frame, (cx, cy), (dcx, dcy), color, 1)
        cv2.circle(frame, (dcx, dcy), 6, color, -1)

        label = f"X {det['conf']:.0%}"
        cv2.putText(frame, label, (int(x1), int(y1) - 8),
                    OVERLAY_FONT, 0.6, color, 2)

        dx_px = det['cx'] - cx
        dy_px = det['cy'] - cy
        cv2.putText(frame, f"dx={dx_px:+.0f} dy={dy_px:+.0f}px",
                    (int(x1), int(y2) + 20), OVERLAY_FONT, 0.5, color, 1)
    elif state not in ("COMMIT_DESCENT", "LAND", "LANDING", "TAKEOFF"):
        cv2.putText(frame, "NO X", (cx - 30, cy + 50),
                    OVERLAY_FONT, 0.8, COLOR_LOST, 2)

    # ── Rolling window progress bar ──────────────────────────
    if latch_info:
        n_centered = latch_info.get('n_centered', 0)
        n_total = latch_info.get('n_total', 0)
        required = latch_info.get('required', LATCH_MIN_CENTERED)
        hit_rate = latch_info.get('hit_rate', 0)

        bar_x, bar_y = 10, h - 70
        bar_w, bar_h = 200, 20

        # Background bar (total window)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), COLOR_LATCH, 1)

        # Fill bar (centered count relative to required)
        fill_frac = min(n_centered / required, 1.0) if required > 0 else 0
        fill_w = int(bar_w * fill_frac)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h), COLOR_LATCH, -1)

        # Text label
        cv2.putText(frame, f"LATCH: {n_centered}/{required} "
                           f"({hit_rate:.0%} hit)",
                    (bar_x, bar_y - 8), OVERLAY_FONT, 0.55, COLOR_LATCH, 2)

        # Mini window visualization (dots showing recent hits/misses)
        dot_y = bar_y + bar_h + 15
        dot_x_start = bar_x
        window_entries = latch_info.get('window_entries', [])
        for i, is_hit in enumerate(window_entries[-40:]):  # show last 40
            dot_x = dot_x_start + i * 6
            color = COLOR_OK if is_hit else (80, 80, 80)
            cv2.circle(frame, (dot_x + 3, dot_y), 2, color, -1)

    # ── Commit descent info ──────────────────────────────────
    if commit_info:
        lat = commit_info.get('lat', 0)
        lon = commit_info.get('lon', 0)
        drift = commit_info.get('drift', 0)
        tgt_alt = commit_info.get('target_alt', 0)

        cv2.drawMarker(frame, (cx, cy), COLOR_COMMIT,
                       cv2.MARKER_CROSS, 60, 3)

        info_lines = [
            f"LATCHED: ({lat:.7f}, {lon:.7f})",
            f"DRIFT: {drift:.2f}m",
            f"TGT ALT: {tgt_alt:.1f}m",
            "VISION OFF - GPS HOLD",
        ]
        for i, line in enumerate(info_lines):
            y_pos = h - 100 + i * 22
            cv2.putText(frame, line, (11, y_pos),
                        OVERLAY_FONT, 0.50, (0, 0, 0), 3)
            cv2.putText(frame, line, (10, y_pos),
                        OVERLAY_FONT, 0.50, COLOR_COMMIT, 1)

    # ── Phase indicator ──────────────────────────────────────
    if state in ("COMMIT_DESCENT", "LAND", "LANDING"):
        phase = "GPS-HOLD"
        phase_color = COLOR_COMMIT
    elif state == "LATCH_CENTERING":
        phase = "LATCHING"
        phase_color = COLOR_LATCH
    else:
        phase = "YOLO"
        phase_color = COLOR_OK
    cv2.putText(frame, f"PHASE: {phase}", (w - 220, 25),
                OVERLAY_FONT, 0.65, phase_color, 2)

    # ── HUD ──────────────────────────────────────────────────
    hud_color = COLOR_OK if det else (COLOR_COMMIT if commit_info else COLOR_LOST)
    lines = [
        f"STATE: {state}",
        f"ALT: {cur_alt:.1f}m",
        f"GPS: {fc.lat:.7f}, {fc.lon:.7f}",
        f"SATS: {fc.satellites}  FIX: {fc.gps_fix}",
        f"BATT: {fc.battery_pct}%",
        f"HDG: {fc.heading:.0f} deg",
    ]
    if centered:
        lines.append("** CENTERED **")

    y_off = 55
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (11, y_off + i * 24),
                    OVERLAY_FONT, 0.55, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, y_off + i * 24),
                    OVERLAY_FONT, 0.55, hud_color, 1)

    # ── Timestamp ────────────────────────────────────────────
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    cv2.putText(frame, ts, (w - 160, h - 12),
                OVERLAY_FONT, 0.5, (255, 255, 255), 1)

    return frame


# ===========================================================================
# HELPER: draw overlay + write video + push to live feed
# ===========================================================================

def process_frame(frame, state, det, cur_alt, fc, vw,
                  centered=False, latch_info=None, commit_info=None):
    if frame is None:
        return 0
    overlay = draw_overlay(frame.copy(), state, det, cur_alt, fc,
                           centered=centered, latch_info=latch_info,
                           commit_info=commit_info)
    if vw:
        vw.write(overlay)
    update_feed_frame(overlay)
    return 1


# ===========================================================================
# CSV DATA LOGGER
# ===========================================================================

def create_csv_log(prefix="flight_data"):
    fname = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    f = open(fname, 'w')
    f.write("timestamp,state,alt_m,lat,lon,gps_fix,satellites,"
            "heading_deg,battery_pct,"
            "yolo_detected,yolo_conf,yolo_cx,yolo_cy,"
            "dx_px,dy_px,fwd_m,right_m,dist_m,"
            "vx_cmd,vy_cmd,vz_cmd,"
            "latch_n_centered,latch_n_total,latch_hit_rate,"
            "latch_lat,latch_lon,"
            "drift_from_latch_m,"
            "frame_num,notes\n")
    return fname, f


def csv_row(f, state, fc, cur_alt, det=None,
            dx_px=0, dy_px=0, fwd_m=0, right_m=0, dist_m=0,
            vx=0, vy=0, vz=0,
            latch_n_centered=0, latch_n_total=0, latch_hit_rate=0,
            latch_lat=0, latch_lon=0,
            drift=0, frame_num=0, notes=""):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    yolo_det = 1 if det else 0
    yolo_conf = det['conf'] if det else 0
    yolo_cx = det['cx'] if det else 0
    yolo_cy = det['cy'] if det else 0
    f.write(f"{ts},{state},{cur_alt:.2f},"
            f"{fc.lat:.8f},{fc.lon:.8f},"
            f"{fc.gps_fix},{fc.satellites},"
            f"{fc.heading:.1f},{fc.battery_pct},"
            f"{yolo_det},{yolo_conf:.3f},{yolo_cx},{yolo_cy},"
            f"{dx_px},{dy_px},{fwd_m:.4f},{right_m:.4f},{dist_m:.4f},"
            f"{vx:.4f},{vy:.4f},{vz:.4f},"
            f"{latch_n_centered},{latch_n_total},{latch_hit_rate:.3f},"
            f"{latch_lat:.8f},{latch_lon:.8f},"
            f"{drift:.4f},{frame_num},{notes}\n")
    f.flush()


# ===========================================================================
# MAIN
# ===========================================================================

def main(args):
    if not args.dry_run and not args.sitl:
        if not confirm("12_land_on_x_rolling_latch.py — ROLLING WINDOW LATCH",
                       f"Takeoff {args.alt}m → Find X (YOLO) → Center → "
                       f"Descend to {LATCH_ALT}m →\n"
                       f"  Rolling window latch ({LATCH_MIN_CENTERED}/"
                       f"{WINDOW_SIZE} centered frames, median GPS) → "
                       f"Commit descent → LAND\n"
                       f"  No vision below {LATCH_ALT}m — pure GPS position hold\n"
                       f"  Video recording: ON\n"
                       f"  Live feed: http://0.0.0.0:{args.feed_port}"):
            return

    # ── Start live feed server ─────────────────────────────────
    start_feed_server(args.feed_port)

    model = load_yolo(args.weights, imgsz=args.imgsz)

    fc = FlightController()
    if not args.dry_run:
        fc.connect()
        if not args.sitl and not fc.preflight():
            fc.close(); return

    cap = open_camera(sitl=args.sitl)
    if not cap and not args.sitl:
        print("[!] No camera — cannot detect X.")
        fc.close(); return

    # ── Video writer ─────────────────────────────────────────
    vw = None
    video_path = None
    video_path_tmp = None
    actual_fps = 20.0
    frame_count = 0
    record_t0 = None

    if cap:
        actual_fps = get_camera_fps(cap, sitl=args.sitl)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path_tmp = f"landing_rolling_{ts}_tmp.mp4"
        video_path = f"landing_rolling_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(video_path_tmp, fourcc, actual_fps,
                             (FRAME_W, FRAME_H))
        if not vw.isOpened():
            print("[!] WARNING: Could not open video writer")
            vw = None
        else:
            print(f"[REC] Recording → {video_path}  ({actual_fps:.1f} FPS)")

    # ── Logs ─────────────────────────────────────────────────
    log_fname, log_f = create_log("landing_rolling")
    csv_fname, csv_f = create_csv_log("landing_rolling_data")

    log(log_f, "=" * 60)
    log(log_f, "ROLLING WINDOW LATCH PRECISION LANDING")
    log(log_f, "=" * 60)
    log(log_f, f"Takeoff alt:       {args.alt}m")
    log(log_f, f"Latch alt:         {LATCH_ALT}m")
    log(log_f, f"Window size:       {WINDOW_SIZE} frames")
    log(log_f, f"Min centered:      {LATCH_MIN_CENTERED} frames")
    log(log_f, f"Latch deadzone:    {LATCH_DEADZONE}px")
    log(log_f, f"Latch max spread:  {LATCH_MAX_SPREAD}m")
    log(log_f, f"Latch timeout:     {LATCH_TIMEOUT}s")
    log(log_f, f"Commit descent:    {COMMIT_VZ} m/s")
    log(log_f, f"Land alt:          {LAND_ALT}m")
    log(log_f, f"YOLO weights:      {args.weights}")
    log(log_f, f"YOLO conf:         {args.conf}")
    log(log_f, f"Camera:            {FRAME_W}x{FRAME_H}")
    log(log_f, f"Live feed port:    {args.feed_port}")
    if video_path:
        log(log_f, f"Video:             {video_path}")
    log(log_f, f"CSV data:          {csv_fname}")
    log(log_f, "")

    with SafeFlight(fc, camera=cap, video_writer=vw) as sf:

        # ── State variables ──────────────────────────────────
        state = "TAKEOFF"
        last_x = 0
        search_t0 = 0
        descend_tgt = 0

        # Rolling window latch
        latch_window = RollingLatchWindow(
            window_size=WINDOW_SIZE,
            min_centered=LATCH_MIN_CENTERED,
            max_spread=LATCH_MAX_SPREAD
        )
        latch_t0 = 0              # when latch phase started
        latch_lat = 0.0
        latch_lon = 0.0
        latch_locked = False

        # Commit state
        commit_t0 = 0
        commit_last_cmd = 0
        commit_target_alt = 0

        if args.dry_run:
            state = "SEARCH"
            log(log_f, "DRY RUN — skipping takeoff")

        # ══════════════════════════════════════════════════════
        # STATE MACHINE
        # ══════════════════════════════════════════════════════
        while state not in ("DONE", "ABORT"):
            if not args.dry_run:
                fc.poll()
            cur_alt = fc.alt if (not args.dry_run and fc.alt > 0.3) else args.alt

            # ── Read camera + detect ─────────────────────────
            det = None
            frame = None
            if cap:
                ret, frame = cap.read()
                if ret and state not in ("COMMIT_DESCENT", "LAND", "LANDING"):
                    det = detect_x(frame, model, args.conf, args.imgsz)
                elif ret:
                    pass
                else:
                    frame = None

            # ── TAKEOFF ──────────────────────────────────────
            if state == "TAKEOFF":
                log(log_f, f"TAKEOFF → {args.alt}m")
                record_t0 = time.time()

                if not fc.set_guided():
                    state = "ABORT"; continue
                if not fc.arm():
                    state = "ABORT"; continue
                if not fc.takeoff(args.alt):
                    fc.set_rtl(); state = "ABORT"; continue
                if not fc.wait_alt(args.alt):
                    fc.set_rtl(); state = "ABORT"; continue

                log(log_f, f"At {fc.alt:.1f}m — stabilizing 3s")

                t0 = time.time()
                while time.time() - t0 < 3:
                    if cap:
                        ret, frm = cap.read()
                        if ret:
                            d = detect_x(frm, model, args.conf, args.imgsz)
                            fc.poll()
                            frame_count += process_frame(frm, "TAKEOFF", d,
                                                          fc.alt, fc, vw)
                    time.sleep(0.05)

                log(log_f, f"GPS at takeoff: ({fc.lat:.8f}, {fc.lon:.8f}) "
                           f"sats={fc.satellites} fix={fc.gps_fix}")
                state = "SEARCH"
                search_t0 = time.time()
                continue

            # ── Write video + stream for all states after takeoff ─
            if frame is not None:
                latch_info_overlay = None
                commit_info_overlay = None
                if state == "LATCH_CENTERING":
                    latch_info_overlay = {
                        'n_centered': latch_window.centered_count,
                        'n_total': latch_window.total_frames,
                        'required': LATCH_MIN_CENTERED,
                        'hit_rate': latch_window.hit_rate,
                        'window_entries': [e[1] for e in latch_window.buffer],
                    }
                if state == "COMMIT_DESCENT" and latch_locked:
                    drift = haversine(fc.lat, fc.lon, latch_lat, latch_lon)
                    commit_info_overlay = {
                        'lat': latch_lat, 'lon': latch_lon,
                        'drift': drift,
                        'target_alt': commit_target_alt,
                    }
                fc_count = process_frame(frame, state, det, cur_alt, fc, vw,
                                         latch_info=latch_info_overlay,
                                         commit_info=commit_info_overlay)
                frame_count += fc_count
                if record_t0 is None and fc_count > 0:
                    record_t0 = time.time()

            # ── SEARCH ───────────────────────────────────────
            if state == "SEARCH":
                if search_t0 == 0:
                    search_t0 = time.time()
                elapsed = time.time() - search_t0

                if det:
                    log(log_f, f"X FOUND conf={det['conf']:.2f} "
                               f"@({det['cx']},{det['cy']}) "
                               f"bbox={det['bbox']}")
                    csv_row(csv_f, state, fc, cur_alt, det,
                            frame_num=frame_count, notes="X_FOUND")
                    state = "ACQUIRE"
                    last_x = time.time()
                    continue

                if elapsed > SEARCH_TIMEOUT:
                    log(log_f, f"SEARCH TIMEOUT ({SEARCH_TIMEOUT}s) → RTL")
                    csv_row(csv_f, state, fc, cur_alt,
                            frame_num=frame_count, notes="SEARCH_TIMEOUT")
                    if not args.dry_run:
                        fc.set_rtl()
                    state = "ABORT"; continue

                csv_row(csv_f, state, fc, cur_alt,
                        frame_num=frame_count, notes=f"searching_{elapsed:.0f}s")
                print(f"\r  [SEARCH] {elapsed:.0f}s / {SEARCH_TIMEOUT:.0f}s | "
                      f"Alt={cur_alt:.1f}m | sats={fc.satellites}   ",
                      end="", flush=True)
                time.sleep(0.05)

            # ── ACQUIRE (YOLO centering) ─────────────────────
            elif state == "ACQUIRE":
                if det is None:
                    lost = time.time() - last_x
                    if lost > LOST_TIMEOUT:
                        log(log_f, f"LOST X {lost:.0f}s → RTL")
                        csv_row(csv_f, state, fc, cur_alt,
                                frame_num=frame_count, notes="LOST_TIMEOUT_RTL")
                        if not args.dry_run:
                            fc.stop(); fc.set_rtl()
                        state = "ABORT"; continue
                    if not args.dry_run:
                        fc.stop()
                    csv_row(csv_f, state, fc, cur_alt,
                            frame_num=frame_count, notes=f"lost_{lost:.1f}s")
                    print(f"\r  [ACQUIRE] Lost X — holding ({lost:.1f}s / "
                          f"{LOST_TIMEOUT:.0f}s)   ", end="", flush=True)
                    time.sleep(VEL_RATE)
                    continue

                last_x = time.time()
                dx_px = det['cx'] - FRAME_W // 2
                dy_px = det['cy'] - FRAME_H // 2

                near_latch = cur_alt < LATCH_ALT + 1.5
                dz = DEADZONE_LATCH if near_latch else DEADZONE_HIGH
                spd = SPEED_LOW if near_latch else SPEED_HIGH

                if abs(dx_px) <= dz and abs(dy_px) <= dz:
                    # ── CENTERED ─────────────────────────────
                    m_fwd, m_right = pixels_to_meters(dx_px, dy_px, cur_alt)
                    dist_m = math.sqrt(m_fwd**2 + m_right**2)

                    log(log_f, f"CENTERED at {cur_alt:.1f}m "
                               f"(offset: {dx_px:+d},{dy_px:+d}px = "
                               f"{m_fwd:+.2f}fwd, {m_right:+.2f}right = "
                               f"{dist_m:.2f}m)")
                    csv_row(csv_f, state, fc, cur_alt, det,
                            dx_px, dy_px, m_fwd, m_right, dist_m,
                            frame_num=frame_count, notes="CENTERED")
                    if not args.dry_run:
                        fc.stop()

                    if frame is not None:
                        frame_count += process_frame(frame, state, det,
                                                      cur_alt, fc, vw,
                                                      centered=True)

                    # Check if we should enter latch phase
                    if cur_alt <= LATCH_ALT + 0.8:
                        state = "LATCH_CENTERING"
                        latch_window.reset()
                        latch_t0 = time.time()
                        log(log_f, f"→ LATCH_CENTERING (alt={cur_alt:.1f}m, "
                                   f"target={LATCH_ALT}m)")
                        log(log_f, f"  Rolling window: {WINDOW_SIZE} frames, "
                                   f"need {LATCH_MIN_CENTERED} centered, "
                                   f"median GPS")
                    else:
                        state = "DESCEND"
                        descend_tgt = max(cur_alt - DESCEND_STEP, LATCH_ALT)
                        log(log_f, f"→ DESCEND to {descend_tgt:.1f}m")
                    time.sleep(0.5)
                    continue

                # ── Not centered — send correction ───────────
                m_fwd, m_right = pixels_to_meters(dx_px, dy_px, cur_alt)
                dist_m = math.sqrt(m_fwd**2 + m_right**2)
                scale = min(spd / dist_m, 1.0) if dist_m > spd else 0.5
                vx = m_fwd * scale
                vy = m_right * scale

                if not args.dry_run:
                    fc.velocity_body(vx, vy, 0)

                csv_row(csv_f, state, fc, cur_alt, det,
                        dx_px, dy_px, m_fwd, m_right, dist_m,
                        vx, vy, 0, frame_num=frame_count)

                parts = []
                if abs(m_fwd) > 0.03:
                    parts.append(f"{'FWD' if m_fwd > 0 else 'BACK'} {abs(m_fwd):.2f}m")
                if abs(m_right) > 0.03:
                    parts.append(f"{'RIGHT' if m_right > 0 else 'LEFT'} {abs(m_right):.2f}m")
                print(f"\r  [ACQUIRE] {' + '.join(parts) or '~'} | "
                      f"v=({vx:.2f},{vy:.2f}) | Alt={cur_alt:.1f}m | "
                      f"conf={det['conf']:.2f} | "
                      f"dz={'TIGHT' if near_latch else 'WIDE'}   ",
                      end="", flush=True)
                time.sleep(VEL_RATE)

            # ── DESCEND (YOLO phase) ─────────────────────────
            elif state == "DESCEND":
                log(log_f, f"DESCEND {cur_alt:.1f}m → {descend_tgt:.1f}m")
                t0 = time.time()
                while True:
                    if not args.dry_run:
                        fc.poll()
                    cur_alt = fc.alt if not args.dry_run else descend_tgt
                    if cur_alt <= descend_tgt + 0.3:
                        break
                    if time.time() - t0 > 15:
                        log(log_f, "DESCEND timeout 15s")
                        break
                    if not args.dry_run:
                        fc.velocity_ned(0, 0, DESCENT_VZ)

                    if cap:
                        ret, frm = cap.read()
                        if ret:
                            d = detect_x(frm, model, args.conf, args.imgsz)
                            if d:
                                last_x = time.time()
                            frame_count += process_frame(frm, "DESCEND", d,
                                                          cur_alt, fc, vw)

                    csv_row(csv_f, "DESCEND", fc, cur_alt,
                            vz=DESCENT_VZ, frame_num=frame_count,
                            notes=f"tgt={descend_tgt:.1f}")
                    print(f"\r  [DESCEND] {cur_alt:.1f}m → {descend_tgt:.1f}m   ",
                          end="", flush=True)
                    time.sleep(VEL_RATE)

                if not args.dry_run:
                    fc.stop()
                log(log_f, f"At {cur_alt:.1f}m — re-acquiring")
                time.sleep(1)
                state = "ACQUIRE"

            # ══════════════════════════════════════════════════
            # ROLLING WINDOW LATCH PHASE
            # ══════════════════════════════════════════════════

            elif state == "LATCH_CENTERING":
                # ── Check latch timeout ──────────────────────
                latch_elapsed = time.time() - latch_t0
                if latch_elapsed > LATCH_TIMEOUT:
                    log(log_f, f"LATCH TIMEOUT ({LATCH_TIMEOUT}s) — "
                               f"window had {latch_window.centered_count}/"
                               f"{latch_window.total_frames} centered → RTL")
                    csv_row(csv_f, state, fc, cur_alt,
                            latch_n_centered=latch_window.centered_count,
                            latch_n_total=latch_window.total_frames,
                            latch_hit_rate=latch_window.hit_rate,
                            frame_num=frame_count, notes="LATCH_TIMEOUT_RTL")
                    if not args.dry_run:
                        fc.stop(); fc.set_rtl()
                    state = "ABORT"; continue

                # ── No detection — add miss to window ────────
                if det is None:
                    lost = time.time() - last_x
                    if lost > LOST_TIMEOUT:
                        log(log_f, f"LOST during latch {lost:.0f}s → RTL")
                        csv_row(csv_f, state, fc, cur_alt,
                                frame_num=frame_count, notes="LATCH_LOST_RTL")
                        if not args.dry_run:
                            fc.stop(); fc.set_rtl()
                        state = "ABORT"; continue
                    if not args.dry_run:
                        fc.stop()
                    # Record as a miss — but DON'T reset the window!
                    # This is the key difference from Script 11.
                    latch_window.add_frame(False, fc.lat, fc.lon)
                    csv_row(csv_f, state, fc, cur_alt,
                            latch_n_centered=latch_window.centered_count,
                            latch_n_total=latch_window.total_frames,
                            latch_hit_rate=latch_window.hit_rate,
                            frame_num=frame_count,
                            notes=f"no_det_lost_{lost:.1f}s")
                    print(f"\r  [LATCH] {latch_window.summary_str()} | "
                          f"Lost X ({lost:.1f}s)   ",
                          end="", flush=True)
                    time.sleep(VEL_RATE)
                    continue

                last_x = time.time()
                dx_px = det['cx'] - FRAME_W // 2
                dy_px = det['cy'] - FRAME_H // 2
                m_fwd, m_right = pixels_to_meters(dx_px, dy_px, cur_alt)
                dist_m = math.sqrt(m_fwd**2 + m_right**2)

                is_centered = (abs(dx_px) <= LATCH_DEADZONE and
                               abs(dy_px) <= LATCH_DEADZONE)

                # ── Add frame to rolling window (hit or miss) ─
                latch_window.add_frame(is_centered, fc.lat, fc.lon,
                                       dx_px, dy_px)

                if is_centered:
                    if not args.dry_run:
                        fc.stop()
                    log(log_f, f"LATCH ● ({dx_px:+d},{dy_px:+d})px "
                               f"{dist_m:.3f}m | "
                               f"GPS=({fc.lat:.8f},{fc.lon:.8f}) | "
                               f"window={latch_window.centered_count}/"
                               f"{latch_window.total_frames} "
                               f"({latch_window.hit_rate:.0%})")

                    # Re-draw with centered flag + latch bar
                    if frame is not None:
                        li = {
                            'n_centered': latch_window.centered_count,
                            'n_total': latch_window.total_frames,
                            'required': LATCH_MIN_CENTERED,
                            'hit_rate': latch_window.hit_rate,
                            'window_entries': [e[1] for e in latch_window.buffer],
                        }
                        frame_count += process_frame(frame, state, det,
                                                      cur_alt, fc, vw,
                                                      centered=True,
                                                      latch_info=li)
                else:
                    # Not centered — correct, but DON'T reset window
                    scale = min(SPEED_LOW / dist_m, 1.0) if dist_m > SPEED_LOW else 0.4
                    vx = m_fwd * scale
                    vy = m_right * scale
                    if not args.dry_run:
                        fc.velocity_body(vx, vy, 0)

                csv_row(csv_f, state, fc, cur_alt, det,
                        dx_px, dy_px, m_fwd, m_right, dist_m,
                        latch_n_centered=latch_window.centered_count,
                        latch_n_total=latch_window.total_frames,
                        latch_hit_rate=latch_window.hit_rate,
                        frame_num=frame_count,
                        notes=f"{'HIT' if is_centered else 'MISS'}")

                print(f"\r  [LATCH] {latch_window.summary_str()} | "
                      f"({dx_px:+d},{dy_px:+d})px | "
                      f"GPS=({fc.lat:.7f},{fc.lon:.7f})   ",
                      end="", flush=True)

                # ── Check if we can latch ────────────────────
                latch_result = latch_window.check_latch()
                if latch_result:
                    # ── LATCH ACCEPTED! ──────────────────────
                    latch_lat = latch_result['lat']
                    latch_lon = latch_result['lon']
                    latch_locked = True

                    log(log_f, "")
                    log(log_f, "=" * 55)
                    log(log_f, "  GPS LATCH — ROLLING WINDOW")
                    log(log_f, "=" * 55)
                    log(log_f, f"  Method:       MEDIAN (outlier-robust)")
                    log(log_f, f"  Window:       {latch_result['n_total']} frames")
                    log(log_f, f"  Centered:     {latch_result['n_centered']} frames "
                               f"({latch_result['hit_rate']:.0%} hit rate)")
                    log(log_f, f"  GPS spread:   {latch_result['spread']:.3f}m "
                               f"(max: {LATCH_MAX_SPREAD}m)")
                    log(log_f, f"  Latch pos:    ({latch_lat:.8f}, {latch_lon:.8f})")
                    log(log_f, f"  Altitude:     {cur_alt:.1f}m")
                    log(log_f, f"  Satellites:   {fc.satellites}  Fix: {fc.gps_fix}")
                    log(log_f, f"  Latch time:   {latch_elapsed:.1f}s")

                    # Log individual readings
                    for i, (lat, lon) in enumerate(latch_result['readings']):
                        d = haversine(lat, lon, latch_lat, latch_lon)
                        log(log_f, f"    [{i+1}] ({lat:.8f}, {lon:.8f}) — "
                                   f"{d:.3f}m from median")

                    log(log_f, "")
                    log(log_f, f"  ✓ GPS LATCHED!")
                    log(log_f, f"  Position: ({latch_lat:.8f}, {latch_lon:.8f})")
                    log(log_f, f"  Vision corrections: OFF from now.")
                    log(log_f, "=" * 55)
                    log(log_f, "")

                    csv_row(csv_f, "LATCHED", fc, cur_alt,
                            latch_n_centered=latch_result['n_centered'],
                            latch_n_total=latch_result['n_total'],
                            latch_hit_rate=latch_result['hit_rate'],
                            latch_lat=latch_lat, latch_lon=latch_lon,
                            frame_num=frame_count,
                            notes=f"LATCHED_spread={latch_result['spread']:.3f}_"
                                  f"time={latch_elapsed:.1f}s")

                    print(f"\n\n  ★ GPS LATCHED at ({latch_lat:.8f}, {latch_lon:.8f})")
                    print(f"    {latch_result['n_centered']}/"
                          f"{latch_result['n_total']} centered | "
                          f"Spread: {latch_result['spread']:.3f}m | "
                          f"Time: {latch_elapsed:.1f}s")
                    print(f"    → Committing descent (no more vision)\n")

                    state = "COMMIT_DESCENT"
                    commit_t0 = time.time()
                    commit_last_cmd = 0
                    commit_target_alt = max(cur_alt - 0.5, LAND_ALT)
                    if not args.dry_run:
                        fc.stop()
                    time.sleep(0.5)
                    continue

                time.sleep(VEL_RATE)

            # ══════════════════════════════════════════════════
            # COMMIT DESCENT (GPS hold, no vision)
            # ══════════════════════════════════════════════════

            elif state == "COMMIT_DESCENT":
                now = time.time()

                drift = haversine(fc.lat, fc.lon, latch_lat, latch_lon)

                elapsed = now - commit_t0
                desired_alt = max(LATCH_ALT - elapsed * COMMIT_VZ, LAND_ALT)
                commit_target_alt = desired_alt

                if now - commit_last_cmd >= COMMIT_CMD_RATE:
                    if not args.dry_run:
                        send_position_with_descent(fc, latch_lat, latch_lon,
                                                   commit_target_alt)
                    commit_last_cmd = now
                    log(log_f, f"COMMIT cmd: hold ({latch_lat:.8f},{latch_lon:.8f}) "
                               f"alt={commit_target_alt:.2f}m | "
                               f"actual={cur_alt:.1f}m | "
                               f"drift={drift:.3f}m | "
                               f"sats={fc.satellites}")

                csv_row(csv_f, state, fc, cur_alt,
                        vz=COMMIT_VZ,
                        latch_lat=latch_lat, latch_lon=latch_lon,
                        drift=drift, frame_num=frame_count,
                        notes=f"tgt_alt={commit_target_alt:.2f}_drift={drift:.3f}")

                if det:
                    dx_px = det['cx'] - FRAME_W // 2
                    dy_px = det['cy'] - FRAME_H // 2
                    m_fwd, m_right = pixels_to_meters(dx_px, dy_px, cur_alt)
                    log(log_f, f"  (YOLO still sees X: conf={det['conf']:.2f} "
                               f"offset=({dx_px:+d},{dy_px:+d})px "
                               f"= ({m_fwd:+.2f}m,{m_right:+.2f}m) "
                               f"— IGNORED, using GPS)")

                print(f"\r  [COMMIT] Alt={cur_alt:.1f}m → "
                      f"{commit_target_alt:.1f}m | "
                      f"drift={drift:.2f}m | "
                      f"sats={fc.satellites} | "
                      f"GPS=({fc.lat:.7f},{fc.lon:.7f})   ",
                      end="", flush=True)

                if cur_alt <= LAND_ALT + 0.3:
                    log(log_f, "")
                    log(log_f, f"Below {LAND_ALT}m (actual={cur_alt:.1f}m) → LAND")
                    log(log_f, f"Final drift from latch: {drift:.3f}m")
                    state = "LAND"
                    continue

                if elapsed > 60:
                    log(log_f, "COMMIT timeout 60s — forcing LAND")
                    state = "LAND"
                    continue

                time.sleep(VEL_RATE)

            # ── LAND ─────────────────────────────────────────
            elif state == "LAND":
                drift = 0
                if latch_locked:
                    drift = haversine(fc.lat, fc.lon, latch_lat, latch_lon)

                log(log_f, "")
                log(log_f, "=" * 55)
                log(log_f, f"LAND MODE at {cur_alt:.1f}m")
                if latch_locked:
                    log(log_f, f"Drift from latch at land trigger: {drift:.3f}m")
                    log(log_f, f"Current GPS: ({fc.lat:.8f}, {fc.lon:.8f})")
                    log(log_f, f"Latch GPS:   ({latch_lat:.8f}, {latch_lon:.8f})")
                log(log_f, "=" * 55)

                csv_row(csv_f, state, fc, cur_alt,
                        latch_lat=latch_lat, latch_lon=latch_lon,
                        drift=drift, frame_num=frame_count,
                        notes="LAND_TRIGGER")

                if not args.dry_run:
                    fc.set_land()

                    land_t0 = time.time()
                    while fc.armed and (time.time() - land_t0 < 30):
                        fc.poll()
                        if cap:
                            ret, frm = cap.read()
                            if ret:
                                drift_now = 0
                                ci = None
                                if latch_locked:
                                    drift_now = haversine(fc.lat, fc.lon,
                                                         latch_lat, latch_lon)
                                    ci = {'lat': latch_lat, 'lon': latch_lon,
                                          'drift': drift_now,
                                          'target_alt': 0}
                                frame_count += process_frame(
                                    frm, "LANDING", None, fc.alt, fc, vw,
                                    commit_info=ci)

                        if int((time.time() - land_t0) * 2) % 2 == 0:
                            d = haversine(fc.lat, fc.lon,
                                          latch_lat, latch_lon) if latch_locked else 0
                            csv_row(csv_f, "LANDING", fc, fc.alt,
                                    latch_lat=latch_lat, latch_lon=latch_lon,
                                    drift=d, frame_num=frame_count,
                                    notes=f"landing_alt={fc.alt:.2f}")
                        time.sleep(0.1)

                # ── Post-landing analysis ────────────────────
                final_drift = 0
                if latch_locked:
                    final_drift = haversine(fc.lat, fc.lon, latch_lat, latch_lon)

                log(log_f, "")
                log(log_f, "=" * 60)
                log(log_f, "  ★ ★ ★  LANDED  ★ ★ ★")
                log(log_f, "=" * 60)
                log(log_f, f"  Final GPS:   ({fc.lat:.8f}, {fc.lon:.8f})")
                if latch_locked:
                    log(log_f, f"  Latch GPS:   ({latch_lat:.8f}, {latch_lon:.8f})")
                    log(log_f, f"  Final drift: {final_drift:.3f}m")
                    log(log_f, f"  Within 1m?   {'YES ✓' if final_drift < 1.0 else 'NO ✗'}")
                    log(log_f, f"  Latch method: Rolling window median")
                else:
                    log(log_f, f"  (No latch — landed without GPS lock)")
                log(log_f, f"  Satellites:  {fc.satellites}")
                log(log_f, f"  GPS fix:     {fc.gps_fix}")
                log(log_f, f"  Battery:     {fc.battery_pct}%")
                log(log_f, "=" * 60)

                csv_row(csv_f, "DONE", fc, fc.alt,
                        latch_lat=latch_lat, latch_lon=latch_lon,
                        drift=final_drift, frame_num=frame_count,
                        notes=f"LANDED_drift={final_drift:.3f}")

                print("\n\n" + "=" * 60)
                print("  ★ ★ ★  LANDED ON X!  ★ ★ ★")
                if latch_locked:
                    print(f"  Final drift from latch: {final_drift:.3f}m")
                    print(f"  Within tolerance: "
                          f"{'YES ✓' if final_drift < 1.0 else 'NO ✗'}")
                print("=" * 60 + "\n")
                state = "DONE"

            # ── ABORT ────────────────────────────────────────
            elif state == "ABORT":
                log(log_f, "ABORTED")
                csv_row(csv_f, "ABORT", fc, cur_alt,
                        frame_num=frame_count, notes="ABORT")
                if not args.dry_run:
                    fc.wait_disarmed(timeout=60)
                state = "DONE"

    # ── Finalize video ────────────────────────────────────────
    if vw:
        vw.release()
        record_elapsed = time.time() - record_t0 if record_t0 else 1
        measured_fps = frame_count / max(record_elapsed, 0.001)
        log(log_f, f"Video: {frame_count} frames, {record_elapsed:.1f}s, "
                   f"measured {measured_fps:.1f} FPS")

        if frame_count > 0 and os.path.isfile(video_path_tmp):
            try:
                import subprocess
                subprocess.run([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", video_path_tmp,
                    "-vf", f"setpts=N/{measured_fps:.2f}/TB",
                    "-r", f"{measured_fps:.2f}",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    video_path
                ], check=True)
                os.remove(video_path_tmp)
                log(log_f, f"Remuxed → {video_path} @ {measured_fps:.1f} FPS")
            except Exception as e:
                os.rename(video_path_tmp, video_path)
                log(log_f, f"ffmpeg remux failed ({e}), raw file kept: {video_path}")
        else:
            if os.path.isfile(video_path_tmp):
                os.rename(video_path_tmp, video_path)

    # ── Close logs ────────────────────────────────────────────
    csv_f.close()
    log_f.close()

    print(f"\n[*] Flight log:  {log_fname}")
    print(f"[*] CSV data:    {csv_fname}")
    if video_path:
        print(f"[*] Video:       {video_path}")
    print(f"[*] Done!")
    print()
    print(f"  TIP: Load the CSV into a spreadsheet or Python to plot")
    print(f"  altitude vs time, drift vs time, and centering error.")
    print(f"  New columns: latch_n_centered, latch_n_total, latch_hit_rate")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Rolling Window Latch Precision X Landing (with live feed)")
    p.add_argument("--alt", type=float, default=TAKEOFF_ALT,
                   help=f"Takeoff altitude (default {TAKEOFF_ALT}m)")
    p.add_argument("--weights", default="best_22.pt",
                   help="YOLO weights file")
    p.add_argument("--conf", type=float, default=0.50,
                   help="YOLO confidence threshold")
    p.add_argument("--imgsz", type=int, default=640,
                   help="YOLO input size")
    p.add_argument("--dry-run", action="store_true",
                   help="Camera only, no flight (test detection)")
    p.add_argument("--sitl", action="store_true",
                   help="SITL mode (webcam)")
    p.add_argument("--feed-port", type=int, default=5000,
                   help="Port for live browser feed (default 5000)")
    main(p.parse_args())
