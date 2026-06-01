#!/usr/bin/env python3
"""
DROP ON X — Package Drop via Rolling Window Latch + LIVE BROWSER FEED
======================================================================
Pivoted from landing mission to PACKAGE DROP mission.

Why?
    Landing requires the drone to descend through turbulent low-altitude
    air and physically touch down on X. In 15+ mph Mojave winds, position
    hold at <2m is too unreliable — GPS drift + wind gusts cause >1m error.

    Package drop only requires centering at 3m and releasing the payload.
    The drone stays at a stable altitude where YOLO can see the full X,
    and the payload falls straight (~0.78s fall from 3m). Much more
    forgiving in wind.

Approach:
    1. Takeoff to cruise altitude (default 5m)
    2. Search for X with YOLO
    3. Center over X, descend in steps to DROP_ALT (3m)
    4. Rolling window latch at 3m — same median-GPS logic as Script 12,
       but with a MUCH larger deadzone (80px ≈ 18cm at 3m instead of 25px)
    5. Once latch triggers → OPEN CLAW (RC channel 6 = 2000)
    6. Hold 2s for payload to fall cleanly
    7. CLOSE CLAW (RC channel 6 = 1000) → RTL

    Key differences from 5_land_on_X_2.py:
    - No descent below 3m — drone stays at drop altitude
    - No GPS blind descent — YOLO stays active the whole time
    - Bigger deadzone (80px not 25px) — wind-tolerant
    - RC channel 6 override for claw (2000=open, 1000=close)
    - RTL after drop instead of LAND

Error budget at 3m drop:
    Centering error:    ~0.10-0.18m (80px deadzone at 3m, median of 6+)
    Fall drift (wind):  ~0.05-0.15m (0.78s fall, package has some inertia)
    Total:              ~0.15-0.33m → well within ±0.5m

State machine:
    TAKEOFF → SEARCH → ACQUIRE → DESCEND → ... →
    DROP_CENTERING (rolling window at DROP_ALT) →
    DROP (open claw) → POST_DROP (close + RTL) → DONE

Usage:
    python3 drop_on_x.py --alt 5
    python3 drop_on_x.py --alt 5 --drop-alt 3
    python3 drop_on_x.py --alt 5 --feed-port 5000
    python3 drop_on_x.py --dry-run

Terminal 1: mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 \\
            --out=udp:127.0.0.1:14551
Terminal 2: python3 drop_on_x.py

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
                          CAM_OFFSET_FWD,
                          confirm, create_log, log)

# ===========================================================================
# TUNABLE PARAMETERS
# ===========================================================================

# ── YOLO centering phase (above DROP_ALT) ────────────────────
DESCEND_STEP     = 1.0       # meters per descent step
DEADZONE_HIGH    = 60        # px — "centered" threshold at cruise alt
DEADZONE_DROP    = 80        # px — "centered" at drop altitude
                             # 80px at 3m ≈ 18cm — vs 25px ≈ 5.7cm before
                             # Your flight 2 data: 12% hit rate at 75px,
                             # with 80px you'll get ~20-30% in wind
SPEED_HIGH       = 0.30      # m/s — centering speed at cruise alt
SPEED_LOW        = 0.15      # m/s — centering speed near drop alt

# ── Drop altitude ────────────────────────────────────────────
DROP_ALT         = 3.0       # meters — altitude to hover and drop payload
                             # At 3m: ground footprint ≈ 4.4m, X fully visible
                             # Fall time: sqrt(2*3/9.81) ≈ 0.78s

# ── Rolling window config (same idea, more forgiving) ────────
# With 80px deadzone at 3m and 15mph gusts, expect 20-40% hit rate.
# 6 out of 30 = 20% minimum → should trigger in 5-15s.
WINDOW_SIZE          = 30    # frames in sliding window (~6s at 5 FPS)
DROP_MIN_CENTERED    = 6     # min centered frames to trigger drop
DROP_MAX_SPREAD      = 2.0   # meters — max GPS spread (wider than landing)
DROP_TIMEOUT         = 45.0  # seconds — max time in drop centering phase
                             # Longer than landing (30s) since we're not
                             # descending, just hovering

# ── Claw control ─────────────────────────────────────────────
CLAW_CHANNEL     = 6         # RC channel for claw servo
CLAW_OPEN_PWM    = 2000      # PWM to open claw (release payload)
CLAW_CLOSE_PWM   = 1000      # PWM to close claw (grip payload)
POST_DROP_HOLD   = 3.0       # seconds to hold position after drop
                             # Lets payload fall clear before RTL

# ── Common ────────────────────────────────────────────────────
LOST_TIMEOUT     = 10.0      # seconds without YOLO detection → RTL
SEARCH_TIMEOUT   = 60.0      # seconds of searching → RTL
VEL_RATE         = 0.2       # seconds between velocity commands
DESCENT_VZ       = 0.30      # m/s — descent rate
ACQUIRE_PATIENCE = 15.0      # seconds — if can't center above drop alt,
                             # descend anyway. Precision only matters at
                             # drop altitude, not above it.
MIN_CORRECT_DIST      = 0.5  # meters — ACQUIRE: skip tiny corrections
MIN_CORRECT_DIST_DROP = 0.2  # meters — DROP_CENTERING: tighter threshold

# ── Blind scoot (camera→claw offset compensation) ────────────
# After drop lock, stop CV, scoot forward by CAM_OFFSET_FWD
# to put the CLAW (not camera) over X, then drop.
BLIND_SCOOT_SPEED = 0.20     # m/s — gentle forward creep
BLIND_SCOOT_HOLD  = 1.0      # seconds — hold after scoot for settling

# ── Video / overlay ───────────────────────────────────────────
OVERLAY_FONT         = cv2.FONT_HERSHEY_SIMPLEX
COLOR_OK             = (0, 255, 0)       # green
COLOR_LOST           = (0, 0, 255)       # red
COLOR_CENTER         = (0, 255, 255)     # yellow
COLOR_DROP_READY     = (0, 165, 255)     # orange — drop centering
COLOR_DROPPED        = (255, 0, 255)     # magenta — payload released


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
    <title>Drop Mission — Live</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #111; color: #eee;
            font-family: -apple-system, system-ui, sans-serif;
            display: flex; flex-direction: column;
            align-items: center; min-height: 100vh;
        }
        h1 { margin: 15px 0 5px; font-size: 1.3em; color: #ff8c00; font-weight: 500; }
        .info { font-size: 0.8em; color: #666; margin-bottom: 10px; }
        .info span { color: #ff8c00; }
        img {
            max-width: 95vw; max-height: 84vh;
            border: 2px solid #333; border-radius: 6px;
        }
    </style>
</head>
<body>
    <h1>📦 Drop Mission — Live Feed</h1>
    <p class="info">Status: <span>STREAMING</span> | Package drop at DROP_ALT meters</p>
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
    """Median GPS position — robust to outliers unlike mean."""
    if not readings:
        return 0.0, 0.0
    lats = [r[0] for r in readings]
    lons = [r[1] for r in readings]
    return statistics.median(lats), statistics.median(lons)


# ===========================================================================
# CLAW CONTROL (RC Channel Override)
# ===========================================================================

def set_claw(fc, pwm, log_f=None):
    """
    Set claw servo via RC channel override.

    Uses MAVLink RC_CHANNELS_OVERRIDE — same as typing 'rc 6 2000'
    in Mission Planner / MAVProxy terminal.

    Args:
        fc: FlightController instance
        pwm: PWM value (2000=open, 1000=close)
        log_f: optional log file handle
    """
    # Build the override: channels 1-5 = 0 (no override), channel 6 = pwm
    # rc_channels_override_send takes channels 1-8
    fc.master.mav.rc_channels_override_send(
        fc.master.target_system,
        fc.master.target_component,
        0,      # ch1 (roll)
        0,      # ch2 (pitch)
        0,      # ch3 (throttle)
        0,      # ch4 (yaw)
        0,      # ch5
        pwm,    # ch6 — CLAW
        0,      # ch7
        0       # ch8
    )
    action = "OPEN" if pwm >= 1500 else "CLOSE"
    msg = f"CLAW {action} — RC{CLAW_CHANNEL}={pwm}"
    print(f"\n  ★ {msg}")
    if log_f:
        log(log_f, msg)


def release_rc_override(fc):
    """Release all RC overrides (set all channels to 0)."""
    fc.master.mav.rc_channels_override_send(
        fc.master.target_system,
        fc.master.target_component,
        0, 0, 0, 0, 0, 0, 0, 0
    )


# ===========================================================================
# ROLLING WINDOW (reused from Script 12, same logic)
# ===========================================================================

class RollingDropWindow:
    """
    Sliding window that tracks recent frames for drop centering.

    Same concept as Script 12's RollingLatchWindow, but tuned for drops:
    - Larger deadzone (80px not 25px)
    - No GPS blind descent after latch — we just open the claw
    """

    def __init__(self, window_size=WINDOW_SIZE,
                 min_centered=DROP_MIN_CENTERED,
                 max_spread=DROP_MAX_SPREAD):
        self.window_size = window_size
        self.min_centered = min_centered
        self.max_spread = max_spread
        self.buffer = deque(maxlen=window_size)

    def add_frame(self, is_centered, lat, lon, dx_px=0, dy_px=0):
        self.buffer.append((time.time(), is_centered, lat, lon, dx_px, dy_px))

    @property
    def total_frames(self):
        return len(self.buffer)

    @property
    def centered_count(self):
        return sum(1 for entry in self.buffer if entry[1])

    @property
    def centered_readings(self):
        return [(entry[2], entry[3]) for entry in self.buffer if entry[1]]

    @property
    def hit_rate(self):
        if not self.buffer:
            return 0.0
        return self.centered_count / len(self.buffer)

    def check_drop_ready(self):
        """
        Check if drop conditions are met.
        Returns None if not ready, dict with drop info if ready.
        """
        n_centered = self.centered_count
        if n_centered < self.min_centered:
            return None

        readings = self.centered_readings
        spread = gps_spread(readings)
        if spread > self.max_spread:
            return None

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
        self.buffer.clear()

    def summary_str(self):
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
                 centered=False, drop_info=None, dropped=False):
    """Draw full HUD overlay on the video frame."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # ── Crosshair ────────────────────────────────────────────
    size = 30
    cv2.line(frame, (cx - size, cy), (cx + size, cy), COLOR_CENTER, 1)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), COLOR_CENTER, 1)

    # ── Deadzone circle ──────────────────────────────────────
    # THIS IS THE KEY FIX: much bigger circle visible on screen
    if state == "DROP_CENTERING":
        dz = DEADZONE_DROP
        dz_color = COLOR_DROP_READY
        # Draw a thick, visible circle so you can see it clearly
        cv2.circle(frame, (cx, cy), dz, dz_color, 2)
        # Also draw an inner reference circle at half size
        cv2.circle(frame, (cx, cy), dz // 2, dz_color, 1)
    elif state == "DROPPED":
        dz = DEADZONE_DROP
        cv2.circle(frame, (cx, cy), dz, COLOR_DROPPED, 3)
    elif cur_alt < DROP_ALT + 1.5:
        dz = DEADZONE_DROP
        cv2.circle(frame, (cx, cy), dz, COLOR_CENTER, 2)
    else:
        dz = DEADZONE_HIGH
        cv2.circle(frame, (cx, cy), dz, COLOR_CENTER, 1)

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
    elif state not in ("DROPPED", "POST_DROP", "TAKEOFF"):
        cv2.putText(frame, "NO X", (cx - 30, cy + 50),
                    OVERLAY_FONT, 0.8, COLOR_LOST, 2)

    # ── Rolling window progress bar ──────────────────────────
    if drop_info:
        n_centered = drop_info.get('n_centered', 0)
        n_total = drop_info.get('n_total', 0)
        required = drop_info.get('required', DROP_MIN_CENTERED)
        hit_rate = drop_info.get('hit_rate', 0)

        bar_x, bar_y = 10, h - 70
        bar_w, bar_h = 200, 20

        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), COLOR_DROP_READY, 1)

        fill_frac = min(n_centered / required, 1.0) if required > 0 else 0
        fill_w = int(bar_w * fill_frac)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h), COLOR_DROP_READY, -1)

        cv2.putText(frame, f"DROP: {n_centered}/{required} "
                           f"({hit_rate:.0%} hit)",
                    (bar_x, bar_y - 8), OVERLAY_FONT, 0.55, COLOR_DROP_READY, 2)

        # Mini window dots
        dot_y = bar_y + bar_h + 15
        dot_x_start = bar_x
        window_entries = drop_info.get('window_entries', [])
        for i, is_hit in enumerate(window_entries[-40:]):
            dot_x = dot_x_start + i * 6
            color = COLOR_OK if is_hit else (80, 80, 80)
            cv2.circle(frame, (dot_x + 3, dot_y), 2, color, -1)

    # ── Drop status banner ───────────────────────────────────
    if dropped:
        cv2.putText(frame, "PAYLOAD RELEASED", (cx - 140, cy + 80),
                    OVERLAY_FONT, 0.9, COLOR_DROPPED, 3)
        cv2.drawMarker(frame, (cx, cy), COLOR_DROPPED,
                       cv2.MARKER_STAR, 60, 3)

    # ── Phase indicator ──────────────────────────────────────
    phase_map = {
        "TAKEOFF": ("TAKEOFF", COLOR_OK),
        "SEARCH": ("YOLO SEARCH", COLOR_OK),
        "ACQUIRE": ("YOLO CENTER", COLOR_OK),
        "DESCEND": ("DESCENDING", COLOR_OK),
        "DROP_CENTERING": ("DROP LOCK", COLOR_DROP_READY),
        "DROP": ("DROPPING!", COLOR_DROPPED),
        "POST_DROP": ("DROP DONE", COLOR_DROPPED),
    }
    phase, phase_color = phase_map.get(state, ("YOLO", COLOR_OK))
    cv2.putText(frame, f"PHASE: {phase}", (w - 250, 25),
                OVERLAY_FONT, 0.65, phase_color, 2)

    # ── HUD ──────────────────────────────────────────────────
    hud_color = COLOR_OK if det else COLOR_LOST
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
                  centered=False, drop_info=None, dropped=False):
    if frame is None:
        return 0
    overlay = draw_overlay(frame.copy(), state, det, cur_alt, fc,
                           centered=centered, drop_info=drop_info,
                           dropped=dropped)
    if vw:
        vw.write(overlay)
    update_feed_frame(overlay)
    return 1


# ===========================================================================
# CSV DATA LOGGER
# ===========================================================================

def create_csv_log(prefix="drop_data"):
    fname = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    f = open(fname, 'w')
    f.write("timestamp,state,alt_m,lat,lon,gps_fix,satellites,"
            "heading_deg,battery_pct,"
            "yolo_detected,yolo_conf,yolo_cx,yolo_cy,"
            "dx_px,dy_px,fwd_m,right_m,dist_m,"
            "vx_cmd,vy_cmd,vz_cmd,"
            "drop_n_centered,drop_n_total,drop_hit_rate,"
            "drop_lat,drop_lon,"
            "frame_num,notes\n")
    return fname, f


def csv_row(f, state, fc, cur_alt, det=None,
            dx_px=0, dy_px=0, fwd_m=0, right_m=0, dist_m=0,
            vx=0, vy=0, vz=0,
            drop_n_centered=0, drop_n_total=0, drop_hit_rate=0,
            drop_lat=0, drop_lon=0,
            frame_num=0, notes=""):
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
            f"{drop_n_centered},{drop_n_total},{drop_hit_rate:.3f},"
            f"{drop_lat:.8f},{drop_lon:.8f},"
            f"{frame_num},{notes}\n")
    f.flush()


# ===========================================================================
# MAIN
# ===========================================================================

def main(args):
    drop_alt = args.drop_alt

    if not args.dry_run and not args.sitl:
        if not confirm("drop_on_x.py — PACKAGE DROP MISSION",
                       f"Takeoff {args.alt}m → Find X (YOLO) → Center → "
                       f"Descend to {drop_alt}m →\n"
                       f"  Rolling window drop lock ({DROP_MIN_CENTERED}/"
                       f"{WINDOW_SIZE} centered, {DEADZONE_DROP}px deadzone) →\n"
                       f"  OPEN CLAW (RC{CLAW_CHANNEL}={CLAW_OPEN_PWM}) → "
                       f"Hold {POST_DROP_HOLD}s → CLOSE → RTL\n"
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
        video_path_tmp = f"drop_mission_{ts}_tmp.mp4"
        video_path = f"drop_mission_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(video_path_tmp, fourcc, actual_fps,
                             (FRAME_W, FRAME_H))
        if not vw.isOpened():
            print("[!] WARNING: Could not open video writer")
            vw = None
        else:
            print(f"[REC] Recording → {video_path}  ({actual_fps:.1f} FPS)")

    # ── Logs ─────────────────────────────────────────────────
    log_fname, log_f = create_log("drop_mission")
    csv_fname, csv_f = create_csv_log("drop_data")

    log(log_f, "=" * 60)
    log(log_f, "PACKAGE DROP MISSION")
    log(log_f, "=" * 60)
    log(log_f, f"Takeoff alt:       {args.alt}m")
    log(log_f, f"Drop alt:          {drop_alt}m")
    log(log_f, f"Deadzone (drop):   {DEADZONE_DROP}px")
    log(log_f, f"Window size:       {WINDOW_SIZE} frames")
    log(log_f, f"Min centered:      {DROP_MIN_CENTERED} frames")
    log(log_f, f"Max GPS spread:    {DROP_MAX_SPREAD}m")
    log(log_f, f"Drop timeout:      {DROP_TIMEOUT}s")
    log(log_f, f"Claw channel:      RC{CLAW_CHANNEL}")
    log(log_f, f"Claw open:         {CLAW_OPEN_PWM}")
    log(log_f, f"Claw close:        {CLAW_CLOSE_PWM}")
    log(log_f, f"Post-drop hold:    {POST_DROP_HOLD}s")
    log(log_f, f"Blind scoot:       {CAM_OFFSET_FWD}m fwd @ {BLIND_SCOOT_SPEED}m/s")
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
        acquire_t0 = 0          # when ACQUIRE started at current altitude

        # Rolling window for drop
        drop_window = RollingDropWindow(
            window_size=WINDOW_SIZE,
            min_centered=DROP_MIN_CENTERED,
            max_spread=DROP_MAX_SPREAD
        )
        drop_t0 = 0
        drop_lat = 0.0
        drop_lon = 0.0
        drop_locked = False
        payload_dropped = False

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
                if ret:
                    # YOLO stays active the ENTIRE time (no blind phase)
                    if state not in ("DROP", "POST_DROP"):
                        det = detect_x(frame, model, args.conf, args.imgsz)
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

                log(log_f, f"At {fc.alt:.1f}m — hovering 10s for EKF yaw alignment")

                t0 = time.time()
                while time.time() - t0 < 10:
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

            # ── Write video + stream for all states ──────────
            if frame is not None:
                drop_info_overlay = None
                if state == "DROP_CENTERING":
                    drop_info_overlay = {
                        'n_centered': drop_window.centered_count,
                        'n_total': drop_window.total_frames,
                        'required': DROP_MIN_CENTERED,
                        'hit_rate': drop_window.hit_rate,
                        'window_entries': [e[1] for e in drop_window.buffer],
                    }
                fc_count = process_frame(frame, state, det, cur_alt, fc, vw,
                                         drop_info=drop_info_overlay,
                                         dropped=payload_dropped)
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
                    acquire_t0 = time.time()
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

                # Use wider deadzone near drop altitude
                near_drop = cur_alt < drop_alt + 1.5
                dz = DEADZONE_DROP if near_drop else DEADZONE_HIGH
                spd = SPEED_LOW if near_drop else SPEED_HIGH

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

                    # Check if we should enter drop centering phase
                    if cur_alt <= drop_alt + 0.8:
                        state = "DROP_CENTERING"
                        drop_window.reset()
                        drop_t0 = time.time()
                        log(log_f, f"→ DROP_CENTERING (alt={cur_alt:.1f}m, "
                                   f"target={drop_alt}m)")
                        log(log_f, f"  Rolling window: {WINDOW_SIZE} frames, "
                                   f"need {DROP_MIN_CENTERED} centered, "
                                   f"deadzone={DEADZONE_DROP}px")
                    else:
                        state = "DESCEND"
                        descend_tgt = max(cur_alt - DESCEND_STEP, drop_alt)
                        log(log_f, f"→ DESCEND to {descend_tgt:.1f}m")
                    time.sleep(0.5)
                    continue

                # ── Not centered — send correction ───────────
                m_fwd, m_right = pixels_to_meters(dx_px, dy_px, cur_alt)
                dist_m = math.sqrt(m_fwd**2 + m_right**2)

                # Skip micro-corrections that wind overwhelms at cruise alt
                if dist_m < MIN_CORRECT_DIST:
                    if not args.dry_run:
                        fc.stop()
                    csv_row(csv_f, state, fc, cur_alt, det,
                            dx_px, dy_px, m_fwd, m_right, dist_m,
                            0, 0, 0, frame_num=frame_count,
                            notes=f"skip_micro_{dist_m:.2f}m")
                    print(f"\r  [ACQUIRE] offset {dist_m:.2f}m < {MIN_CORRECT_DIST}m "
                          f"— holding | Alt={cur_alt:.1f}m   ",
                          end="", flush=True)
                    time.sleep(VEL_RATE)
                    continue

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
                      f"dz={'DROP' if near_drop else 'WIDE'}   ",
                      end="", flush=True)

                # ── Patience timer: can't center? descend anyway ──
                # Above drop alt, precision doesn't matter — just get
                # down to drop altitude where the rolling window will
                # handle the final centering. Don't waste battery
                # fighting wind at cruise altitude.
                acquire_elapsed = time.time() - acquire_t0
                if (acquire_elapsed > ACQUIRE_PATIENCE and
                        cur_alt > drop_alt + 0.8):
                    log(log_f, f"PATIENCE EXPIRED ({acquire_elapsed:.0f}s) — "
                               f"X visible but not centered, descending anyway")
                    log(log_f, f"  Current offset: ({dx_px:+d},{dy_px:+d})px "
                               f"= {dist_m:.2f}m")
                    csv_row(csv_f, state, fc, cur_alt, det,
                            dx_px, dy_px, m_fwd, m_right, dist_m,
                            frame_num=frame_count,
                            notes=f"PATIENCE_{acquire_elapsed:.0f}s")
                    if not args.dry_run:
                        fc.stop()
                    state = "DESCEND"
                    descend_tgt = max(cur_alt - DESCEND_STEP, drop_alt)
                    log(log_f, f"→ DESCEND to {descend_tgt:.1f}m (forced)")
                    time.sleep(0.5)
                    continue

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
                acquire_t0 = time.time()
                state = "ACQUIRE"

            # ══════════════════════════════════════════════════
            # DROP CENTERING (rolling window at drop altitude)
            # ══════════════════════════════════════════════════

            elif state == "DROP_CENTERING":
                drop_elapsed = time.time() - drop_t0
                if drop_elapsed > DROP_TIMEOUT:
                    log(log_f, f"DROP TIMEOUT ({DROP_TIMEOUT}s) — "
                               f"window had {drop_window.centered_count}/"
                               f"{drop_window.total_frames} centered → RTL")
                    csv_row(csv_f, state, fc, cur_alt,
                            drop_n_centered=drop_window.centered_count,
                            drop_n_total=drop_window.total_frames,
                            drop_hit_rate=drop_window.hit_rate,
                            frame_num=frame_count, notes="DROP_TIMEOUT_RTL")
                    if not args.dry_run:
                        fc.stop(); fc.set_rtl()
                    state = "ABORT"; continue

                # ── No detection — add miss ──────────────────
                if det is None:
                    lost = time.time() - last_x
                    if lost > LOST_TIMEOUT:
                        log(log_f, f"LOST during drop centering {lost:.0f}s → RTL")
                        csv_row(csv_f, state, fc, cur_alt,
                                frame_num=frame_count, notes="DROP_LOST_RTL")
                        if not args.dry_run:
                            fc.stop(); fc.set_rtl()
                        state = "ABORT"; continue
                    if not args.dry_run:
                        fc.stop()
                    drop_window.add_frame(False, fc.lat, fc.lon)
                    csv_row(csv_f, state, fc, cur_alt,
                            drop_n_centered=drop_window.centered_count,
                            drop_n_total=drop_window.total_frames,
                            drop_hit_rate=drop_window.hit_rate,
                            frame_num=frame_count,
                            notes=f"no_det_lost_{lost:.1f}s")
                    print(f"\r  [DROP] {drop_window.summary_str()} | "
                          f"Lost X ({lost:.1f}s)   ",
                          end="", flush=True)
                    time.sleep(VEL_RATE)
                    continue

                last_x = time.time()
                dx_px = det['cx'] - FRAME_W // 2
                dy_px = det['cy'] - FRAME_H // 2
                m_fwd, m_right = pixels_to_meters(dx_px, dy_px, cur_alt)
                dist_m = math.sqrt(m_fwd**2 + m_right**2)

                is_centered = (abs(dx_px) <= DEADZONE_DROP and
                               abs(dy_px) <= DEADZONE_DROP)

                drop_window.add_frame(is_centered, fc.lat, fc.lon,
                                       dx_px, dy_px)

                if is_centered:
                    if not args.dry_run:
                        fc.stop()
                    log(log_f, f"DROP ● ({dx_px:+d},{dy_px:+d})px "
                               f"{dist_m:.3f}m | "
                               f"GPS=({fc.lat:.8f},{fc.lon:.8f}) | "
                               f"window={drop_window.centered_count}/"
                               f"{drop_window.total_frames} "
                               f"({drop_window.hit_rate:.0%})")

                    if frame is not None:
                        di = {
                            'n_centered': drop_window.centered_count,
                            'n_total': drop_window.total_frames,
                            'required': DROP_MIN_CENTERED,
                            'hit_rate': drop_window.hit_rate,
                            'window_entries': [e[1] for e in drop_window.buffer],
                        }
                        frame_count += process_frame(frame, state, det,
                                                      cur_alt, fc, vw,
                                                      centered=True,
                                                      drop_info=di)
                else:
                    # Not centered — correct position, don't reset window
                    if dist_m >= MIN_CORRECT_DIST_DROP:
                        scale = min(SPEED_LOW / dist_m, 1.0) if dist_m > SPEED_LOW else 0.4
                        vx = m_fwd * scale
                        vy = m_right * scale
                        if not args.dry_run:
                            fc.velocity_body(vx, vy, 0)
                    else:
                        if not args.dry_run:
                            fc.stop()

                csv_row(csv_f, state, fc, cur_alt, det,
                        dx_px, dy_px, m_fwd, m_right, dist_m,
                        drop_n_centered=drop_window.centered_count,
                        drop_n_total=drop_window.total_frames,
                        drop_hit_rate=drop_window.hit_rate,
                        frame_num=frame_count,
                        notes=f"{'HIT' if is_centered else 'MISS'}")

                print(f"\r  [DROP] {drop_window.summary_str()} | "
                      f"({dx_px:+d},{dy_px:+d})px | "
                      f"GPS=({fc.lat:.7f},{fc.lon:.7f})   ",
                      end="", flush=True)

                # ── Check if we can drop ─────────────────────
                drop_result = drop_window.check_drop_ready()
                if drop_result:
                    # ══════════════════════════════════════════
                    # DROP CONFIRMED! OPEN THE CLAW!
                    # ══════════════════════════════════════════
                    drop_lat = drop_result['lat']
                    drop_lon = drop_result['lon']
                    drop_locked = True

                    log(log_f, "")
                    log(log_f, "=" * 55)
                    log(log_f, "  ★ ★ ★  DROP LOCK CONFIRMED  ★ ★ ★")
                    log(log_f, "=" * 55)
                    log(log_f, f"  Method:       MEDIAN (rolling window)")
                    log(log_f, f"  Window:       {drop_result['n_total']} frames")
                    log(log_f, f"  Centered:     {drop_result['n_centered']} frames "
                               f"({drop_result['hit_rate']:.0%} hit rate)")
                    log(log_f, f"  GPS spread:   {drop_result['spread']:.3f}m "
                               f"(max: {DROP_MAX_SPREAD}m)")
                    log(log_f, f"  Drop pos:     ({drop_lat:.8f}, {drop_lon:.8f})")
                    log(log_f, f"  Altitude:     {cur_alt:.1f}m")
                    log(log_f, f"  Satellites:   {fc.satellites}  Fix: {fc.gps_fix}")
                    log(log_f, f"  Lock time:    {drop_elapsed:.1f}s")
                    log(log_f, "=" * 55)

                    csv_row(csv_f, "DROP_LOCKED", fc, cur_alt,
                            drop_n_centered=drop_result['n_centered'],
                            drop_n_total=drop_result['n_total'],
                            drop_hit_rate=drop_result['hit_rate'],
                            drop_lat=drop_lat, drop_lon=drop_lon,
                            frame_num=frame_count,
                            notes=f"LOCKED_spread={drop_result['spread']:.3f}_"
                                  f"time={drop_elapsed:.1f}s")

                    print(f"\n\n  ★ DROP LOCKED at ({drop_lat:.8f}, {drop_lon:.8f})")
                    print(f"    {drop_result['n_centered']}/"
                          f"{drop_result['n_total']} centered | "
                          f"Spread: {drop_result['spread']:.3f}m | "
                          f"Time: {drop_elapsed:.1f}s")
                    print(f"    → SCOOT FWD {CAM_OFFSET_FWD}m then DROP!\n")

                    state = "DROP"
                    if not args.dry_run:
                        fc.stop()
                    time.sleep(0.3)
                    continue

                time.sleep(VEL_RATE)

            # ══════════════════════════════════════════════════
            # DROP — BLIND SCOOT + OPEN CLAW
            # ══════════════════════════════════════════════════

            elif state == "DROP":
                # ── BLIND SCOOT: camera is over X, move fwd ──
                # Camera is centered on X but claw is CAM_OFFSET_FWD
                # (0.38m) behind. Scoot forward blindly to put
                # the claw directly over X, then drop.
                scoot_dist = 1.2
                scoot_time = scoot_dist / BLIND_SCOOT_SPEED

                log(log_f, "")
                log(log_f, "=" * 55)
                log(log_f, f"  → BLIND SCOOT: {scoot_dist:.2f}m fwd "
                           f"@ {BLIND_SCOOT_SPEED:.2f}m/s "
                           f"({scoot_time:.1f}s)")
                log(log_f, "=" * 55)

                if not args.dry_run:
                    scoot_t0 = time.time()
                    while time.time() - scoot_t0 < scoot_time:
                        fc.velocity_body(BLIND_SCOOT_SPEED, 0, 0)
                        fc.poll()
                        # Keep recording video during scoot
                        if cap:
                            ret, frm = cap.read()
                            if ret:
                                frame_count += process_frame(
                                    frm, "DROP", None,
                                    fc.alt, fc, vw, dropped=False)
                        time.sleep(VEL_RATE)
                    fc.stop()

                log(log_f, f"  Scoot done — holding {BLIND_SCOOT_HOLD}s to settle")
                if not args.dry_run:
                    time.sleep(BLIND_SCOOT_HOLD)

                csv_row(csv_f, "SCOOT_DONE", fc, cur_alt,
                        drop_lat=drop_lat, drop_lon=drop_lon,
                        frame_num=frame_count,
                        notes=f"scoot_{scoot_dist:.2f}m_fwd")

                # ── NOW OPEN THE CLAW ────────────────────────
                log(log_f, "")
                log(log_f, "=" * 55)
                log(log_f, "  📦  OPENING CLAW — PAYLOAD AWAY!")
                log(log_f, "=" * 55)

                # OPEN THE CLAW
                if not args.dry_run:
                    set_claw(fc, CLAW_OPEN_PWM, log_f)
                    # Send the override a few times to make sure it sticks
                    for _ in range(5):
                        set_claw(fc, CLAW_OPEN_PWM)
                        time.sleep(0.1)
                else:
                    log(log_f, f"DRY RUN: would send RC{CLAW_CHANNEL}={CLAW_OPEN_PWM}")

                payload_dropped = True
                csv_row(csv_f, "DROP", fc, cur_alt,
                        drop_lat=drop_lat, drop_lon=drop_lon,
                        frame_num=frame_count,
                        notes=f"CLAW_OPEN_RC{CLAW_CHANNEL}={CLAW_OPEN_PWM}")

                # Hold position while payload falls
                log(log_f, f"Holding position {POST_DROP_HOLD}s for payload to clear...")
                hold_t0 = time.time()
                while time.time() - hold_t0 < POST_DROP_HOLD:
                    if not args.dry_run:
                        fc.poll()
                    if cap:
                        ret, frm = cap.read()
                        if ret:
                            frame_count += process_frame(frm, "DROP", None,
                                                          fc.alt if not args.dry_run else cur_alt,
                                                          fc, vw, dropped=True)
                    # Keep sending claw-open to be safe
                    if not args.dry_run:
                        set_claw(fc, CLAW_OPEN_PWM)
                    time.sleep(0.2)

                state = "POST_DROP"
                continue

            # ══════════════════════════════════════════════════
            # POST_DROP — CLOSE CLAW AND RTL
            # ══════════════════════════════════════════════════

            elif state == "POST_DROP":
                log(log_f, "Closing claw and returning home")

                # CLOSE CLAW
                if not args.dry_run:
                    set_claw(fc, CLAW_CLOSE_PWM, log_f)
                    time.sleep(0.5)
                    # Release RC override so RTL controls normally
                    release_rc_override(fc)
                    time.sleep(0.3)

                csv_row(csv_f, "POST_DROP", fc, cur_alt,
                        drop_lat=drop_lat, drop_lon=drop_lon,
                        frame_num=frame_count,
                        notes=f"CLAW_CLOSED_RC{CLAW_CHANNEL}={CLAW_CLOSE_PWM}")

                # RTL
                if not args.dry_run:
                    fc.set_rtl()
                    log(log_f, "RTL commanded")

                    # Wait for disarm
                    rtl_t0 = time.time()
                    while fc.armed and (time.time() - rtl_t0 < 120):
                        fc.poll()
                        if cap:
                            ret, frm = cap.read()
                            if ret:
                                frame_count += process_frame(frm, "RTL", None,
                                                              fc.alt, fc, vw,
                                                              dropped=True)
                        time.sleep(0.2)

                # ── Post-drop report ─────────────────────────
                log(log_f, "")
                log(log_f, "=" * 60)
                log(log_f, "  ★ ★ ★  MISSION COMPLETE  ★ ★ ★")
                log(log_f, "=" * 60)
                log(log_f, f"  Drop GPS:    ({drop_lat:.8f}, {drop_lon:.8f})")
                log(log_f, f"  Drop alt:    {drop_alt:.1f}m")
                log(log_f, f"  Final GPS:   ({fc.lat:.8f}, {fc.lon:.8f})")
                if drop_locked:
                    drift = haversine(fc.lat, fc.lon, drop_lat, drop_lon)
                    log(log_f, f"  RTL drift:   {drift:.3f}m (from drop point)")
                log(log_f, f"  Satellites:  {fc.satellites}")
                log(log_f, f"  Battery:     {fc.battery_pct}%")
                log(log_f, "=" * 60)

                csv_row(csv_f, "DONE", fc, fc.alt if not args.dry_run else 0,
                        drop_lat=drop_lat, drop_lon=drop_lon,
                        frame_num=frame_count,
                        notes="MISSION_COMPLETE")

                print("\n\n" + "=" * 60)
                print("  ★ ★ ★  PACKAGE DROPPED!  ★ ★ ★")
                if drop_locked:
                    print(f"  Drop position: ({drop_lat:.8f}, {drop_lon:.8f})")
                print("=" * 60 + "\n")
                state = "DONE"

            # ── ABORT ────────────────────────────────────────
            elif state == "ABORT":
                log(log_f, "ABORTED — closing claw for safety")
                if not args.dry_run:
                    set_claw(fc, CLAW_CLOSE_PWM, log_f)
                    time.sleep(0.5)
                    release_rc_override(fc)
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
    print(f"  altitude vs time and centering error over the drop phase.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Package Drop Mission — Center over X and release payload")
    p.add_argument("--alt", type=float, default=TAKEOFF_ALT,
                   help=f"Takeoff/cruise altitude (default {TAKEOFF_ALT}m)")
    p.add_argument("--drop-alt", type=float, default=DROP_ALT,
                   help=f"Altitude to hover and drop (default {DROP_ALT}m)")
    p.add_argument("--weights", default="best_22.pt",
                   help="YOLO weights file")
    p.add_argument("--conf", type=float, default=0.75,
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
