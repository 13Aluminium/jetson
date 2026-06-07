#!/usr/bin/env python3
"""
mother.py — Full Autonomous Delivery Mission
==============================================
The "mother script" that chains everything together:

    1. LAUNCH to cruise altitude
    2. TRANSIT to GPS coordinates at configurable speed
    3. SEARCH for X — multi-phase search pattern:
       a) VISUAL SCAN — look straight ahead (direction of travel)
       b) ROTATION SCAN — 360° yaw sweep (4 × 90° CCW, 2s dwell each stop)
       c) ALTITUDE STEP — climb +2m and repeat rotation scan (up to --max-alt)
       d) CROSS SWEEP — 6-leg cross: fwd/back axis, left/right axis
          Always yaws to face travel direction before each leg
       e) If all fail → RTL
    4. ACQUIRING — YOLO centering over X
    5. DESCENDING — step down to drop altitude
    6. DROP ALIGNMENT — rolling window latch
    7. DROPPING — open claw, release payload
    8. RETURNING — close claw, RTL home

    On any failure, claw opens to drop payload before RTL.
    Pilot can trigger RTL from RC at any time.

Phase Names:
    LAUNCH            Takeoff to cruise altitude
    TRANSIT           Fly to target GPS coordinates
    VISUAL SCAN       Look straight ahead at arrival heading
    ROTATION SCAN     360° yaw sweep (4 × 90° CCW, 2s dwell each)
    ALTITUDE STEP     Climb +2m for another rotation scan
    CROSS SWEEP       6-leg cross pattern: fwd/back axis then left/right axis
                      Always yaws to face travel direction before each leg
    ACQUIRING         YOLO centering corrections over X
    DESCENDING        Step-down altitude toward drop height
    DROP ALIGNMENT    Rolling window latch at drop altitude
    DROPPING          Claw open, payload release
    RETURNING         Claw close, RTL home

State Machine:
    LAUNCH → TRANSIT → VISUAL_SCAN → ROTATION_SCAN →
    (ALTITUDE_STEP → ROTATION_SCAN)* → CROSS_SWEEP →
    ACQUIRING → DESCENDING → DROP_ALIGNMENT → DROPPING → RETURNING → DONE

Usage:
    python3 mother.py --lat 33.78310 --lon -118.10940

    python3 mother.py --lat 33.78310 --lon -118.10940 --speed 0.5 \\
                      --alt 5 --max-alt 15 --drop-alt 3

    python3 mother.py --lat 33.78310 --lon -118.10940 --cross-dist 5

    python3 mother.py --lat 33.78310 --lon -118.10940 --dry-run

Terminal 1: mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 \\
            --out=udp:127.0.0.1:14551
Terminal 2: python3 mother.py --lat <lat> --lon <lon>

Failsafes:
    Ctrl+C → RTL | Exception → RTL | Pilot RC → RTL
    X lost 10s during centering → RTL
    Battery failsafe handled by Pixhawk firmware.
    ALL failure paths open claw (drop payload) before RTL —
    the drone never returns with payload still attached.
"""

import argparse
import math
import os
import sys
import time
import cv2
import threading
import statistics
from collections import deque
from datetime import datetime

from pymavlink import mavutil
from flask import Flask, Response, render_template_string

from flight_utils import (
    FlightController, SafeFlight, open_camera, load_yolo, detect_x,
    pixels_to_meters, get_camera_fps,
    TAKEOFF_ALT, FRAME_W, FRAME_H, CAM_OFFSET_FWD,
    confirm, create_log, log,
)


# ===========================================================================
# TUNABLE PARAMETERS
# ===========================================================================

# ── Navigation ────────────────────────────────────────────────
DEFAULT_SPEED     = 0.5     # m/s cruise to target
ARRIVE_RADIUS     = 1.5     # meters — "arrived" at target GPS
NAV_TIMEOUT       = 300     # seconds max for GPS navigation leg

# ── Search pattern ────────────────────────────────────────────
YAW_STEP_DEG      = 90      # degrees per yaw step in scan
YAW_SPEED         = 20      # °/s rotation speed
YAW_SETTLE        = 1.0     # seconds to settle after yaw before scanning
YAW_SCAN_DWELL    = 2.0     # seconds to look for X at each 90° stop
SEARCH_DWELL      = 3.0     # seconds for initial visual scan (straight ahead)
ALT_CLIMB_STEP    = 2.0     # meters to climb between scan rounds
CROSS_PROBE_DIST  = 3.0     # meters for cross-sweep legs
CROSS_PROBE_SPEED = 0.5     # m/s for cross-sweep movement
HEADING_TOL       = 5.0     # degrees — yaw "done" tolerance

# ── YOLO centering (from script 11) ──────────────────────────
DESCEND_STEP      = 1.0     # meters per descent step
DEADZONE_HIGH     = 60      # px — "centered" at cruise alt
DEADZONE_DROP     = 80      # px — "centered" at drop alt
SPEED_HIGH        = 0.30    # m/s centering speed at cruise
SPEED_LOW         = 0.15    # m/s centering speed near drop
DROP_ALT_DEFAULT  = 3.0     # meters — default drop altitude

# ── Rolling window for drop ──────────────────────────────────
WINDOW_SIZE           = 30
DROP_MIN_CENTERED     = 6
DROP_MAX_SPREAD       = 2.0    # meters
DROP_TIMEOUT          = 45.0   # seconds

# ── Claw ──────────────────────────────────────────────────────
CLAW_CHANNEL      = 6
CLAW_OPEN_PWM     = 1000
CLAW_CLOSE_PWM    = 1550
POST_DROP_HOLD    = 3.0     # seconds after claw open

# ── Safety ────────────────────────────────────────────────────
LOST_TIMEOUT       = 10.0   # seconds without YOLO → RTL during centering
VEL_RATE           = 0.2    # seconds between velocity commands
DESCENT_VZ         = 0.30   # m/s descent rate
ACQUIRE_PATIENCE   = 15.0   # seconds to try centering before forced descend
SAFE_RTL_ALT       = 5.0    # meters — climb to this alt before any RTL
MIN_CORRECT_DIST      = 0.5  # meters — ACQUIRING: skip tiny corrections
MIN_CORRECT_DIST_DROP = 0.2  # meters — DROP_ALIGNMENT: tighter threshold

# ── Blind scoot (camera→claw offset compensation) ────────────
# After drop lock, stop CV, scoot forward by offset to put the
# CLAW (not camera) over X, then drop.
BLIND_SCOOT_DIST  = 1.2      # meters — total forward scoot
BLIND_SCOOT_SPEED = 0.20     # m/s — gentle forward creep
BLIND_SCOOT_HOLD  = 1.0      # seconds — hold after scoot for settling

# ── Video / overlay ──────────────────────────────────────────
OVERLAY_FONT      = cv2.FONT_HERSHEY_SIMPLEX
COLOR_OK          = (0, 255, 0)
COLOR_LOST        = (0, 0, 255)
COLOR_CENTER      = (0, 255, 255)
COLOR_DROP_READY  = (0, 165, 255)
COLOR_DROPPED     = (255, 0, 255)
COLOR_NAV         = (255, 200, 0)
FEED_QUALITY      = 60


# ===========================================================================
# LIVE FEED (Flask MJPEG server)
# ===========================================================================
latest_frame = None
frame_lock = threading.Lock()
flask_app = Flask(__name__)

FEED_HTML = """<!DOCTYPE html><html><head><title>Mother Mission</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{background:#111;color:#eee;
font-family:system-ui;display:flex;flex-direction:column;align-items:center;
min-height:100vh}h1{margin:15px 0 5px;font-size:1.3em;color:#ff8c00}
.info{font-size:.8em;color:#666;margin-bottom:10px}.info span{color:#ff8c00}
img{max-width:95vw;max-height:84vh;border:2px solid #333;border-radius:6px}
</style></head><body><h1>Mother Mission — Live Feed</h1>
<p class="info">Status: <span>STREAMING</span></p>
<img src="/video_feed" alt="Live Feed"></body></html>"""


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
            fb = latest_frame
        if fb is None:
            time.sleep(0.05)
            continue
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + fb + b'\r\n'
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
    print(f"{'='*55}\n")
    t = threading.Thread(
        target=lambda: flask_app.run(host='0.0.0.0', port=port, threaded=True,
                                     use_reloader=False),
        daemon=True)
    t.start()
    return t


# ===========================================================================
# GPS HELPERS
# ===========================================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lat1, lon1, lat2, lon2):
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(rlat2)
    y = (math.cos(rlat1) * math.sin(rlat2) -
         math.sin(rlat1) * math.cos(rlat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def gps_spread(readings):
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
    if not readings:
        return 0.0, 0.0
    lats = [r[0] for r in readings]
    lons = [r[1] for r in readings]
    return statistics.median(lats), statistics.median(lons)


def bearing_to_compass(deg):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[round(deg / 22.5) % 16]


# ===========================================================================
# CLAW CONTROL
# ===========================================================================

def set_claw(fc, pwm, log_f=None):
    fc.master.mav.rc_channels_override_send(
        fc.master.target_system, fc.master.target_component,
        0, 0, 0, 0, 0, pwm, 0, 0)
    action = "OPEN" if pwm == CLAW_OPEN_PWM else "CLOSE"
    msg = f"CLAW {action} — RC{CLAW_CHANNEL}={pwm}"
    print(f"\n  ★ {msg}")
    if log_f:
        log(log_f, msg)


def release_rc_override(fc):
    fc.master.mav.rc_channels_override_send(
        fc.master.target_system, fc.master.target_component,
        0, 0, 0, 0, 0, 0, 0, 0)


def safe_rtl(fc, log_f, cap=None, vw=None, frame_count_ref=None,
             mission_t0=None, safe_alt=SAFE_RTL_ALT, drop_first=False):
    """Climb to safe altitude before commanding RTL.

    Prevents low-altitude RTL that could crash into obstacles.
    If drop_first=True, opens the claw to release payload before RTL
    (never land with payload still attached on a failed mission).
    """
    if drop_first:
        log(log_f, "EMERGENCY DROP — opening claw before RTL")
        set_claw(fc, CLAW_OPEN_PWM, log_f)
        for _ in range(5):
            set_claw(fc, CLAW_OPEN_PWM)
            time.sleep(0.1)
        hold_t0 = time.time()
        while time.time() - hold_t0 < POST_DROP_HOLD:
            fc.poll()
            set_claw(fc, CLAW_OPEN_PWM)
            time.sleep(0.2)
        log(log_f, f"Payload released — held {POST_DROP_HOLD}s")

    set_claw(fc, CLAW_CLOSE_PWM, log_f)
    time.sleep(0.3)
    release_rc_override(fc)
    time.sleep(0.2)

    fc.poll()
    cur = fc.alt
    if cur < safe_alt - 0.5:
        log(log_f, f"SAFE RTL: climbing {cur:.1f}m → {safe_alt:.1f}m before RTL")
        t0 = time.time()
        while time.time() - t0 < 20:               # 20s safety timeout
            fc.poll()
            if fc.alt >= safe_alt - 0.3:
                break
            fc.velocity_ned(0, 0, -0.5)             # climb 0.5 m/s
            # keep recording video while climbing
            if cap and vw and frame_count_ref is not None:
                ret, frm = cap.read()
                if ret:
                    elapsed = time.time() - mission_t0 if mission_t0 else 0
                    frame_count_ref[0] += process_frame(
                        frm, "CLIMBING_RTL", None, fc.alt, fc, vw,
                        mission_elapsed=elapsed,
                        extra_info=f"Climbing to {safe_alt:.0f}m for RTL")
            time.sleep(0.2)
        fc.stop()
        time.sleep(0.5)
        log(log_f, f"SAFE RTL: at {fc.alt:.1f}m — commanding RTL")
    else:
        log(log_f, f"SAFE RTL: already at {cur:.1f}m — commanding RTL")

    fc.set_rtl()
    log(log_f, "RTL commanded")


# ===========================================================================
# ROLLING WINDOW (from script 11)
# ===========================================================================

class RollingDropWindow:
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
        return sum(1 for e in self.buffer if e[1])

    @property
    def centered_readings(self):
        return [(e[2], e[3]) for e in self.buffer if e[1]]

    @property
    def hit_rate(self):
        return self.centered_count / len(self.buffer) if self.buffer else 0.0

    def check_drop_ready(self):
        n = self.centered_count
        if n < self.min_centered:
            return None
        readings = self.centered_readings
        spread = gps_spread(readings)
        if spread > self.max_spread:
            return None
        lat, lon = median_gps(readings)
        return {
            'lat': lat, 'lon': lon, 'spread': spread,
            'n_centered': n, 'n_total': len(self.buffer),
            'hit_rate': self.hit_rate, 'readings': readings,
        }

    def reset(self):
        self.buffer.clear()

    def summary_str(self):
        n = self.centered_count
        total = len(self.buffer)
        vis = "".join("●" if e[1] else "○" for e in self.buffer)
        return f"{vis} {n}/{total} ({self.hit_rate:.0%})"


# ===========================================================================
# VIDEO OVERLAY
# ===========================================================================

def draw_overlay(frame, state, det, cur_alt, fc,
                 centered=False, drop_info=None, dropped=False,
                 mission_elapsed=0, extra_info=""):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Crosshair
    cv2.line(frame, (cx - 30, cy), (cx + 30, cy), COLOR_CENTER, 1)
    cv2.line(frame, (cx, cy - 30), (cx, cy + 30), COLOR_CENTER, 1)

    # Deadzone circle
    if state == "DROP_ALIGNMENT":
        cv2.circle(frame, (cx, cy), DEADZONE_DROP, COLOR_DROP_READY, 2)
        cv2.circle(frame, (cx, cy), DEADZONE_DROP // 2, COLOR_DROP_READY, 1)
    elif state in ("DROPPING", "RETURNING"):
        cv2.circle(frame, (cx, cy), DEADZONE_DROP, COLOR_DROPPED, 3)
    elif cur_alt < DROP_ALT_DEFAULT + 1.5:
        cv2.circle(frame, (cx, cy), DEADZONE_DROP, COLOR_CENTER, 2)
    else:
        cv2.circle(frame, (cx, cy), DEADZONE_HIGH, COLOR_CENTER, 1)

    # YOLO detection box
    if det:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLOR_OK, 2)
        dcx, dcy = int(det['cx']), int(det['cy'])
        cv2.line(frame, (cx, cy), (dcx, dcy), COLOR_OK, 1)
        cv2.circle(frame, (dcx, dcy), 6, COLOR_OK, -1)
        cv2.putText(frame, f"X {det['conf']:.0%}",
                    (int(x1), int(y1) - 8), OVERLAY_FONT, 0.6, COLOR_OK, 2)
    elif state not in ("LAUNCH", "TRANSIT", "DROPPING", "RETURNING"):
        cv2.putText(frame, "NO X", (cx - 30, cy + 50),
                    OVERLAY_FONT, 0.8, COLOR_LOST, 2)

    # Rolling window bar
    if drop_info:
        n_c = drop_info.get('n_centered', 0)
        req = drop_info.get('required', DROP_MIN_CENTERED)
        hr = drop_info.get('hit_rate', 0)
        bar_x, bar_y, bar_w, bar_h = 10, h - 70, 200, 20
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), COLOR_DROP_READY, 1)
        fill = int(bar_w * min(n_c / req, 1.0)) if req > 0 else 0
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill, bar_y + bar_h), COLOR_DROP_READY, -1)
        cv2.putText(frame, f"DROP: {n_c}/{req} ({hr:.0%})",
                    (bar_x, bar_y - 8), OVERLAY_FONT, 0.55, COLOR_DROP_READY, 2)

    if dropped:
        cv2.putText(frame, "PAYLOAD RELEASED", (cx - 140, cy + 80),
                    OVERLAY_FONT, 0.9, COLOR_DROPPED, 3)

    # Phase indicator
    phase_colors = {
        "LAUNCH": COLOR_OK, "TRANSIT": COLOR_NAV,
        "VISUAL_SCAN": COLOR_CENTER, "ROTATION_SCAN": COLOR_CENTER,
        "ALTITUDE_STEP": COLOR_CENTER, "CROSS_SWEEP": COLOR_CENTER,
        "ACQUIRING": COLOR_OK, "DESCENDING": COLOR_OK,
        "DROP_ALIGNMENT": COLOR_DROP_READY,
        "DROPPING": COLOR_DROPPED, "RETURNING": COLOR_DROPPED,
    }
    pc = phase_colors.get(state, COLOR_OK)
    cv2.putText(frame, f"PHASE: {state}", (w - 320, 25),
                OVERLAY_FONT, 0.6, pc, 2)

    # Mission elapsed timer
    timer_color = COLOR_OK
    mins = int(mission_elapsed) // 60
    secs = int(mission_elapsed) % 60
    cv2.putText(frame, f"T+{mins}:{secs:02d}", (w - 120, 55),
                OVERLAY_FONT, 0.6, timer_color, 2)

    # HUD
    hud_color = COLOR_OK if det else COLOR_LOST
    lines = [
        f"STATE: {state}",
        f"ALT: {cur_alt:.1f}m",
        f"GPS: {fc.lat:.7f}, {fc.lon:.7f}",
        f"SATS: {fc.satellites}  FIX: {fc.gps_fix}",
        f"BATT: {fc.battery_pct}%",
        f"HDG: {fc.heading:.0f}°",
    ]
    if extra_info:
        lines.append(extra_info)
    if centered:
        lines.append("** CENTERED **")

    y_off = 55
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (11, y_off + i * 22),
                    OVERLAY_FONT, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, y_off + i * 22),
                    OVERLAY_FONT, 0.5, hud_color, 1)

    # Timestamp
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    cv2.putText(frame, ts, (w - 160, h - 12), OVERLAY_FONT, 0.5, (255, 255, 255), 1)
    return frame


def process_frame(frame, state, det, cur_alt, fc, vw,
                  centered=False, drop_info=None, dropped=False,
                  mission_elapsed=0, extra_info=""):
    if frame is None:
        return 0
    overlay = draw_overlay(frame.copy(), state, det, cur_alt, fc,
                           centered=centered, drop_info=drop_info,
                           dropped=dropped, mission_elapsed=mission_elapsed,
                           extra_info=extra_info)
    if vw:
        vw.write(overlay)
    update_feed_frame(overlay)
    return 1


# ===========================================================================
# CSV DATA LOGGER
# ===========================================================================

def create_csv_log(prefix="mother_data"):
    fname = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    f = open(fname, 'w')
    f.write("timestamp,state,alt_m,lat,lon,gps_fix,satellites,"
            "heading_deg,battery_pct,"
            "yolo_detected,yolo_conf,yolo_cx,yolo_cy,"
            "dx_px,dy_px,fwd_m,right_m,dist_m,"
            "vx_cmd,vy_cmd,vz_cmd,"
            "drop_n_centered,drop_n_total,drop_hit_rate,"
            "drop_lat,drop_lon,"
            "mission_elapsed,frame_num,notes\n")
    return fname, f


def csv_row(f, state, fc, cur_alt, det=None,
            dx_px=0, dy_px=0, fwd_m=0, right_m=0, dist_m=0,
            vx=0, vy=0, vz=0,
            drop_n_centered=0, drop_n_total=0, drop_hit_rate=0,
            drop_lat=0, drop_lon=0,
            mission_elapsed=0, frame_num=0, notes=""):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    f.write(f"{ts},{state},{cur_alt:.2f},"
            f"{fc.lat:.8f},{fc.lon:.8f},{fc.gps_fix},{fc.satellites},"
            f"{fc.heading:.1f},{fc.battery_pct},"
            f"{1 if det else 0},{det['conf'] if det else 0:.3f},"
            f"{det['cx'] if det else 0},{det['cy'] if det else 0},"
            f"{dx_px},{dy_px},{fwd_m:.4f},{right_m:.4f},{dist_m:.4f},"
            f"{vx:.4f},{vy:.4f},{vz:.4f},"
            f"{drop_n_centered},{drop_n_total},{drop_hit_rate:.3f},"
            f"{drop_lat:.8f},{drop_lon:.8f},"
            f"{mission_elapsed:.1f},{frame_num},{notes}\n")
    f.flush()


# ===========================================================================
# NAVIGATION HELPERS
# ===========================================================================

def send_goto(fc, lat, lon, alt):
    """Send a GUIDED goto command to a global GPS coordinate."""
    fc.master.mav.set_position_target_global_int_send(
        0, fc.master.target_system, fc.master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0b0000_1111_1111_1000,
        int(lat * 1e7), int(lon * 1e7), alt,
        0, 0, 0, 0, 0, 0, 0, 0)


def send_yaw(fc, angle_deg, speed_deg_s, direction, relative=True):
    """Yaw command. direction: 1=CW, -1=CCW."""
    fc.master.mav.command_long_send(
        fc.master.target_system, fc.master.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0,
        abs(angle_deg), speed_deg_s, direction,
        1 if relative else 0, 0, 0, 0)


def set_speed(fc, speed_mps):
    """Set cruise speed via DO_CHANGE_SPEED."""
    fc.master.mav.command_long_send(
        fc.master.target_system, fc.master.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 0,
        0, speed_mps, -1, 0, 0, 0, 0)


def normalize_heading(h):
    return h % 360


def wait_yaw_complete(fc, target_heading, timeout=30):
    """Wait for heading to reach target. Returns True on success."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        fc.poll()
        diff = abs(fc.heading - target_heading)
        if diff > 180:
            diff = 360 - diff
        if diff <= HEADING_TOL:
            return True
        time.sleep(0.1)
    return False


def scan_for_x(cap, model, conf, imgsz, dwell_time, fc, vw,
               state_name, mission_t0, frame_count_ref, log_f=None):
    """
    Look for X via YOLO for dwell_time seconds.
    Returns the detection dict if found, else None.
    Also updates video feed and frame counter.
    """
    t0 = time.time()
    while time.time() - t0 < dwell_time:
        if cap:
            ret, frame = cap.read()
            if ret:
                det = detect_x(frame, model, conf, imgsz)
                fc.poll()
                elapsed = time.time() - mission_t0
                frame_count_ref[0] += process_frame(
                    frame, state_name, det, fc.alt, fc, vw,
                    mission_elapsed=elapsed,
                    extra_info=f"Scanning... {time.time()-t0:.1f}s")
                if det:
                    if log_f:
                        log(log_f, f"X FOUND during {state_name} conf={det['conf']:.2f}")
                    return det
        time.sleep(0.05)
    return None


def move_body_and_scan(fc, cap, model, conf, imgsz, fwd, right,
                       speed, vw, mission_t0, frame_count_ref, log_f):
    """
    Move in body frame by (fwd, right) meters at given speed,
    scanning for X the entire time. Returns detection if found.
    """
    dist = math.sqrt(fwd**2 + right**2)
    if dist < 0.01:
        return None
    travel_time = dist / speed
    # Normalize to unit vector scaled by speed
    vx = (fwd / dist) * speed
    vy = (right / dist) * speed

    t0 = time.time()
    while time.time() - t0 < travel_time:
        fc.poll()
        fc.velocity_body(vx, vy, 0)

        if cap:
            ret, frame = cap.read()
            if ret:
                det = detect_x(frame, model, conf, imgsz)
                elapsed = time.time() - mission_t0
                frame_count_ref[0] += process_frame(
                    frame, "CROSS_SWEEP", det, fc.alt, fc, vw,
                    mission_elapsed=elapsed,
                    extra_info=f"Probing fwd={fwd:+.0f} right={right:+.0f}")
                if det:
                    fc.stop()
                    log(log_f, f"X FOUND during cross probe! conf={det['conf']:.2f}")
                    return det
        time.sleep(VEL_RATE)

    fc.stop()
    time.sleep(0.5)  # settle
    return None


# ===========================================================================
# MAIN
# ===========================================================================

def main(args):
    global CROSS_PROBE_DIST
    CROSS_PROBE_DIST = args.cross_dist

    drop_alt = args.drop_alt
    target_lat = args.lat
    target_lon = args.lon

    # ── Sanity checks ─────────────────────────────────────────
    if args.alt < drop_alt:
        print(f"[!] ERROR: Cruise alt ({args.alt}m) must be >= drop alt ({drop_alt}m)")
        sys.exit(1)
    if args.max_alt < args.alt:
        print(f"[!] ERROR: Max alt ({args.max_alt}m) must be >= cruise alt ({args.alt}m)")
        sys.exit(1)

    # ── Confirmation ──────────────────────────────────────────
    if not args.dry_run and not args.sitl:
        dist_est = "unknown"
        desc = (
            f"FULL DELIVERY MISSION\n"
            f"  Target GPS:  ({target_lat:.8f}, {target_lon:.8f})\n"
            f"  Cruise alt:  {args.alt}m  |  Max alt: {args.max_alt}m  |  Drop alt: {drop_alt}m\n"
            f"  Speed:       {args.speed} m/s\n"
            f"  Cross sweep: {CROSS_PROBE_DIST:.1f}m legs\n"
            f"  Plan: Takeoff → Fly to GPS → Search for X → Center → Drop payload → RTL"
        )
        if not confirm("mother.py — DELIVERY MISSION", desc):
            return

    # ── Start live feed ───────────────────────────────────────
    start_feed_server(args.feed_port)

    # ── YOLO model ────────────────────────────────────────────
    model = load_yolo(args.weights, imgsz=args.imgsz)

    # ── Flight controller ─────────────────────────────────────
    fc = FlightController()
    if not args.dry_run:
        fc.connect()
        if not args.sitl and not fc.preflight():
            fc.close()
            return

    # ── Camera ────────────────────────────────────────────────
    cap = open_camera(sitl=args.sitl)
    if not cap and not args.sitl:
        print("[!] No camera — cannot detect X.")
        fc.close()
        return

    # ── Video writer ──────────────────────────────────────────
    vw = None
    video_path = None
    video_path_tmp = None
    actual_fps = 20.0
    frame_count = [0]  # mutable ref for helpers
    record_t0 = None

    if cap:
        actual_fps = get_camera_fps(cap, sitl=args.sitl)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path_tmp = f"mother_mission_{ts_str}_tmp.mp4"
        video_path = f"mother_mission_{ts_str}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(video_path_tmp, fourcc, actual_fps, (FRAME_W, FRAME_H))
        if not vw.isOpened():
            print("[!] WARNING: Could not open video writer")
            vw = None
        else:
            print(f"[REC] Recording → {video_path} ({actual_fps:.1f} FPS)")

    # ── Logs ──────────────────────────────────────────────────
    log_fname, log_f = create_log("mother_mission")
    csv_fname, csv_f = create_csv_log("mother_data")

    log(log_f, "=" * 60)
    log(log_f, "MOTHER MISSION — FULL AUTONOMOUS DELIVERY")
    log(log_f, "=" * 60)
    log(log_f, f"Target GPS:      ({target_lat:.8f}, {target_lon:.8f})")
    log(log_f, f"Cruise alt:      {args.alt}m")
    log(log_f, f"Max alt:         {args.max_alt}m")
    log(log_f, f"Drop alt:        {drop_alt}m")
    log(log_f, f"Speed:           {args.speed} m/s")
    log(log_f, f"Cross sweep:     {CROSS_PROBE_DIST:.1f}m legs")
    log(log_f, f"YOLO weights:    {args.weights}")
    log(log_f, f"YOLO conf:       {args.conf}")
    log(log_f, f"Blind scoot:     {BLIND_SCOOT_DIST}m fwd @ {BLIND_SCOOT_SPEED}m/s")
    log(log_f, f"Min correct:     {MIN_CORRECT_DIST}m (cruise) / {MIN_CORRECT_DIST_DROP}m (drop)")
    log(log_f, "")

    with SafeFlight(fc, camera=cap, video_writer=vw) as sf:

        # ── State variables ───────────────────────────────────
        state = "LAUNCH"
        mission_t0 = time.time()
        last_x = 0
        acquire_t0 = 0
        descend_tgt = 0
        yaw_scan_step = 0       # 0-3 for 4 × 90° CCW
        search_alt = args.alt   # current search altitude

        # Cross-pattern state
        cross_phase = 0         # 0=fwd, 1=back, 2=left, 3=right

        # Drop state
        drop_window = RollingDropWindow(
            window_size=WINDOW_SIZE,
            min_centered=DROP_MIN_CENTERED,
            max_spread=DROP_MAX_SPREAD)
        drop_t0 = 0
        drop_lat = 0.0
        drop_lon = 0.0
        drop_locked = False
        payload_dropped = False

        # Navigation state
        initial_dist = 0
        travel_heading = 0      # heading when arriving at target

        if args.dry_run:
            state = "VISUAL_SCAN"
            log(log_f, "DRY RUN — skipping takeoff and navigation")

        # ══════════════════════════════════════════════════════
        # STATE MACHINE
        # ══════════════════════════════════════════════════════
        while state not in ("DONE", "ABORT"):

            # ── Track elapsed time (for logging/overlay) ─────
            mission_elapsed = time.time() - mission_t0

            # ── Poll FC ──────────────────────────────────────
            if not args.dry_run:
                fc.poll()

            cur_alt = fc.alt if (not args.dry_run and fc.alt > 0.3) else args.alt

            # ── Read camera (background for non-search states) ──
            det = None
            frame = None
            if cap and state not in ("LAUNCH", "TRANSIT", "DROPPING", "RETURNING"):
                ret, frame = cap.read()
                if ret:
                    det = detect_x(frame, model, args.conf, args.imgsz)
                else:
                    frame = None

            # ── Always process frame for video/feed (matches script 11) ──
            # This keeps the camera buffer drained and the live feed
            # updated even when a state handler doesn't process the frame
            # (e.g., ACQUIRING when det is None). Without this, the
            # camera buffer can go stale on Jetson, causing repeated
            # identical frames and missed detections.
            if frame is not None and state not in ("LAUNCH", "TRANSIT"):
                drop_info_overlay = None
                if state == "DROP_ALIGNMENT":
                    drop_info_overlay = {
                        'n_centered': drop_window.centered_count,
                        'n_total': drop_window.total_frames,
                        'required': DROP_MIN_CENTERED,
                        'hit_rate': drop_window.hit_rate,
                        'window_entries': [e[1] for e in drop_window.buffer],
                    }
                frame_count[0] += process_frame(
                    frame, state, det, cur_alt, fc, vw,
                    drop_info=drop_info_overlay,
                    dropped=payload_dropped,
                    mission_elapsed=time.time() - mission_t0)

            # ══════════════════════════════════════════════════
            # LAUNCH
            # ══════════════════════════════════════════════════
            if state == "LAUNCH":
                log(log_f, f"LAUNCH → {args.alt}m")
                record_t0 = time.time()

                if not fc.set_guided():
                    state = "ABORT"; continue
                if not fc.arm():
                    state = "ABORT"; continue
                if not fc.takeoff(args.alt):
                    safe_rtl(fc, log_f, cap, vw, frame_count, mission_t0)
                    state = "ABORT"; continue
                if not fc.wait_alt(args.alt):
                    safe_rtl(fc, log_f, cap, vw, frame_count, mission_t0)
                    state = "ABORT"; continue

                log(log_f, f"At {fc.alt:.1f}m — stabilizing 5s for EKF alignment")
                t0 = time.time()
                while time.time() - t0 < 5:
                    if cap:
                        ret, frm = cap.read()
                        if ret:
                            fc.poll()
                            frame_count[0] += process_frame(
                                frm, "LAUNCH", None, fc.alt, fc, vw,
                                mission_elapsed=time.time() - mission_t0)
                    time.sleep(0.05)

                # Calculate distance and bearing to target
                initial_dist = haversine(fc.lat, fc.lon, target_lat, target_lon)
                travel_heading = bearing(fc.lat, fc.lon, target_lat, target_lon)
                compass = bearing_to_compass(travel_heading)

                log(log_f, f"GPS: ({fc.lat:.8f}, {fc.lon:.8f}) sats={fc.satellites}")
                log(log_f, f"Target: ({target_lat:.8f}, {target_lon:.8f})")
                log(log_f, f"Distance: {initial_dist:.1f}m  Bearing: {travel_heading:.1f}° ({compass})")

                # Set cruise speed
                set_speed(fc, args.speed)
                log(log_f, f"Speed set to {args.speed} m/s")

                state = "TRANSIT"
                continue

            # ══════════════════════════════════════════════════
            # FLY TO TARGET GPS
            # ══════════════════════════════════════════════════
            elif state == "TRANSIT":
                log(log_f, f"NAVIGATING to ({target_lat:.8f}, {target_lon:.8f}) "
                           f"at {args.speed} m/s")

                nav_t0 = time.time()
                last_cmd = 0

                while True:
                    fc.poll()
                    now = time.time()
                    mission_elapsed = now - mission_t0

                    # Resend goto every 2s
                    if now - last_cmd >= 2.0:
                        send_goto(fc, target_lat, target_lon, args.alt)
                        last_cmd = now

                    remaining = haversine(fc.lat, fc.lon, target_lat, target_lon)
                    pct = max(0, (1 - remaining / initial_dist)) * 100 if initial_dist > 0 else 100

                    # Record video during flight
                    if cap:
                        ret, frm = cap.read()
                        if ret:
                            frame_count[0] += process_frame(
                                frm, "TRANSIT", None, fc.alt, fc, vw,
                                mission_elapsed=mission_elapsed,
                                extra_info=f"rem={remaining:.1f}m {pct:.0f}%")

                    csv_row(csv_f, state, fc, fc.alt,
                            mission_elapsed=mission_elapsed,
                            frame_num=frame_count[0],
                            notes=f"remaining={remaining:.1f}m")

                    print(f"\r  [NAV] {pct:5.1f}%  rem={remaining:.1f}m  "
                          f"alt={fc.alt:.1f}m  sats={fc.satellites}  "
                          f"batt={fc.battery_pct}%  T+{mission_elapsed:.0f}s   ",
                          end="", flush=True)

                    if remaining <= ARRIVE_RADIUS:
                        print()
                        log(log_f, f"ARRIVED at target — {remaining:.2f}m away")
                        break

                    if now - nav_t0 > NAV_TIMEOUT:
                        print()
                        log(log_f, f"NAV TIMEOUT {NAV_TIMEOUT}s — searching from here")
                        break

                    time.sleep(0.5)

                if state == "ABORT":
                    continue

                fc.stop()
                # Record the heading we arrived on (direction of travel)
                fc.poll()
                travel_heading = fc.heading
                log(log_f, f"Arrival heading: {travel_heading:.1f}° — will search this direction first")

                # Stabilize 2s at target
                t0 = time.time()
                while time.time() - t0 < 2:
                    fc.poll()
                    time.sleep(0.2)

                state = "VISUAL_SCAN"
                search_alt = fc.alt if not args.dry_run else args.alt
                continue

            # ══════════════════════════════════════════════════
            # VISUAL SCAN — look straight ahead
            # ══════════════════════════════════════════════════
            elif state == "VISUAL_SCAN":
                log(log_f, f"VISUAL SCAN — looking ahead (heading {fc.heading:.0f}°) "
                           f"for {SEARCH_DWELL}s")

                found = scan_for_x(cap, model, args.conf, args.imgsz,
                                   SEARCH_DWELL, fc, vw, "VISUAL_SCAN",
                                   mission_t0, frame_count, log_f)
                if found:
                    det = found
                    last_x = time.time()
                    acquire_t0 = time.time()
                    state = "ACQUIRING"
                    log(log_f, f"X found in visual scan! → ACQUIRING")
                    continue

                log(log_f, "X not found straight ahead → YAW SEARCH (360° scan)")
                yaw_scan_step = 0
                state = "ROTATION_SCAN"
                continue

            # ══════════════════════════════════════════════════
            # YAW SEARCH — 4 × 90° CCW rotation scan
            # ══════════════════════════════════════════════════
            elif state == "ROTATION_SCAN":
                if yaw_scan_step >= 4:
                    # Full 360° done — no X found
                    log(log_f, f"360° scan complete at alt={cur_alt:.1f}m — no X")

                    # Can we climb?
                    next_alt = search_alt + ALT_CLIMB_STEP
                    if next_alt <= args.max_alt:
                        search_alt = next_alt
                        state = "ALTITUDE_STEP"
                        log(log_f, f"Climbing to {search_alt:.1f}m for another scan")
                    else:
                        log(log_f, f"Already at max alt ({args.max_alt}m) — trying cross pattern")
                        cross_phase = 0
                        state = "CROSS_SWEEP"
                    continue

                # Do a 90° CCW rotation
                step_num = yaw_scan_step + 1
                log(log_f, f"ROTATION SCAN step {step_num}/4 — rotating 90° CCW")

                target_hdg = normalize_heading(fc.heading + YAW_STEP_DEG)
                if not args.dry_run:
                    send_yaw(fc, YAW_STEP_DEG, YAW_SPEED, 1, relative=True)
                    timeout = max(YAW_STEP_DEG / YAW_SPEED * 2, 15)
                    wait_yaw_complete(fc, target_hdg, timeout=timeout)

                # Settle after rotation
                time.sleep(YAW_SETTLE)

                # Dwell 2 seconds scanning for X at this heading
                log(log_f, f"Scanning at heading {fc.heading:.0f}° for {YAW_SCAN_DWELL}s")
                found = scan_for_x(cap, model, args.conf, args.imgsz,
                                   YAW_SCAN_DWELL, fc, vw, "ROTATION_SCAN",
                                   mission_t0, frame_count, log_f)
                if found:
                    det = found
                    last_x = time.time()
                    acquire_t0 = time.time()
                    state = "ACQUIRING"
                    log(log_f, f"X found during rotation scan step {step_num}! → ACQUIRING")
                    continue

                yaw_scan_step += 1
                csv_row(csv_f, state, fc, cur_alt,
                        mission_elapsed=time.time() - mission_t0,
                        frame_num=frame_count[0],
                        notes=f"yaw_step_{step_num}_no_x")
                continue

            # ══════════════════════════════════════════════════
            # ALTITUDE STEP — go higher for another 360° scan
            # ══════════════════════════════════════════════════
            elif state == "ALTITUDE_STEP":
                log(log_f, f"CLIMBING {cur_alt:.1f}m → {search_alt:.1f}m")

                if not args.dry_run:
                    # Use velocity to climb
                    t0 = time.time()
                    while True:
                        fc.poll()
                        if fc.alt >= search_alt - 0.5:
                            break
                        if time.time() - t0 > 20:
                            log(log_f, "Climb timeout 20s")
                            break
                        fc.velocity_ned(0, 0, -0.5)  # climb = negative vz in NED

                        if cap:
                            ret, frm = cap.read()
                            if ret:
                                d = detect_x(frm, model, args.conf, args.imgsz)
                                frame_count[0] += process_frame(
                                    frm, "ALTITUDE_STEP", d, fc.alt, fc, vw,
                                    mission_elapsed=time.time() - mission_t0,
                                    extra_info=f"Climbing to {search_alt:.0f}m")
                                if d:
                                    fc.stop()
                                    det = d
                                    last_x = time.time()
                                    acquire_t0 = time.time()
                                    state = "ACQUIRING"
                                    log(log_f, f"X found while climbing! → ACQUIRING")
                                    break
                        time.sleep(VEL_RATE)

                    if state == "ACQUIRING":
                        continue
                    fc.stop()

                log(log_f, f"At {fc.alt:.1f}m — starting new 360° scan")
                time.sleep(1)
                yaw_scan_step = 0
                state = "ROTATION_SCAN"
                continue

            # ══════════════════════════════════════════════════
            # CROSS SEARCH — probe fwd/back/left/right 3m
            # ══════════════════════════════════════════════════
            elif state == "CROSS_SWEEP":
                # ── 6-LEG CROSS PATTERN ──────────────────────
                # Always yaw to face travel direction, then move forward.
                # Uses CROSS_PROBE_DIST (D) for leg lengths.
                # Leg 0: face H,        fwd Dm   → arrive (0, +D)
                # Leg 1: yaw 180°,      fwd 2Dm  → arrive (0, -D)  [passes origin]
                # Leg 2: yaw 180°,      fwd Dm   → arrive (0,  0)  [back to origin]
                # Leg 3: yaw 90° CCW,   fwd Dm   → arrive (-D, 0)
                # Leg 4: yaw 180°,      fwd 2Dm  → arrive (+D, 0)  [passes origin]
                # Leg 5: yaw 180°,      fwd Dm   → arrive (0,  0)  [back to origin]

                #           (yaw_angle, yaw_direction, move_dist, label)
                D = CROSS_PROBE_DIST
                cross_legs = [
                    (0,    0,  D,     f"FWD {D:.0f}m to (0,+{D:.0f})"),
                    (180,  1,  D * 2, f"FLIP+FWD {D*2:.0f}m to (0,-{D:.0f})"),
                    (180,  1,  D,     f"FLIP+FWD {D:.0f}m back to origin"),
                    (90,  -1,  D,     f"CCW 90°+FWD {D:.0f}m to (-{D:.0f},0)"),
                    (180,  1,  D * 2, f"FLIP+FWD {D*2:.0f}m to (+{D:.0f},0)"),
                    (180,  1,  D,     f"FLIP+FWD {D:.0f}m back to origin"),
                ]

                if cross_phase >= len(cross_legs):
                    log(log_f, "CROSS SWEEP EXHAUSTED — X not found → DROPPING & RTL")
                    csv_row(csv_f, state, fc, cur_alt,
                            mission_elapsed=time.time() - mission_t0,
                            frame_num=frame_count[0], notes="CROSS_EXHAUSTED_RTL")
                    if not args.dry_run:
                        fc.stop()
                        safe_rtl(fc, log_f, cap, vw, frame_count, mission_t0,
                                 drop_first=True)
                    state = "ABORT"
                    continue

                yaw_angle, yaw_dir, move_dist, label = cross_legs[cross_phase]
                leg_num = cross_phase + 1
                log(log_f, f"CROSS SWEEP leg {leg_num}/6: {label}")

                # ── Yaw to face travel direction (if needed) ──
                if yaw_angle > 0 and not args.dry_run:
                    target_hdg = normalize_heading(
                        fc.heading + (yaw_angle * yaw_dir if yaw_dir != 0 else 0))
                    # yaw_dir: 1=CW, -1=CCW
                    send_yaw(fc, yaw_angle, YAW_SPEED, yaw_dir, relative=True)
                    yaw_timeout = max(yaw_angle / YAW_SPEED * 2, 10)
                    wait_yaw_complete(fc, target_hdg, timeout=yaw_timeout)
                    time.sleep(YAW_SETTLE)
                    log(log_f, f"  Yaw complete — heading {fc.heading:.0f}°")

                # ── Move forward while scanning ───────────────
                if not args.dry_run:
                    found = move_body_and_scan(
                        fc, cap, model, args.conf, args.imgsz,
                        move_dist, 0.0, CROSS_PROBE_SPEED,
                        vw, mission_t0, frame_count, log_f)
                    if found:
                        det = found
                        last_x = time.time()
                        acquire_t0 = time.time()
                        state = "ACQUIRING"
                        log(log_f, f"X found during cross sweep leg {leg_num}! → ACQUIRING")
                        continue

                cross_phase += 1
                csv_row(csv_f, state, fc, cur_alt,
                        mission_elapsed=time.time() - mission_t0,
                        frame_num=frame_count[0],
                        notes=f"cross_leg_{leg_num}_no_x")
                continue

            # ══════════════════════════════════════════════════
            # ACQUIRE — YOLO centering over X
            # ══════════════════════════════════════════════════
            elif state == "ACQUIRING":
                if det is None:
                    lost = time.time() - last_x
                    if lost > LOST_TIMEOUT:
                        log(log_f, f"LOST X {lost:.0f}s → DROPPING & RTL")
                        csv_row(csv_f, state, fc, cur_alt,
                                mission_elapsed=time.time() - mission_t0,
                                frame_num=frame_count[0], notes="LOST_TIMEOUT_RTL")
                        if not args.dry_run:
                            fc.stop()
                            safe_rtl(fc, log_f, cap, vw, frame_count, mission_t0,
                                     drop_first=True)
                        state = "ABORT"; continue
                    if not args.dry_run:
                        fc.stop()
                    print(f"\r  [ACQUIRE] Lost X — holding ({lost:.1f}s / "
                          f"{LOST_TIMEOUT:.0f}s)   ", end="", flush=True)
                    time.sleep(VEL_RATE)
                    continue

                last_x = time.time()
                dx_px = det['cx'] - FRAME_W // 2
                dy_px = det['cy'] - FRAME_H // 2

                near_drop = cur_alt < drop_alt + 1.5
                dz = DEADZONE_DROP if near_drop else DEADZONE_HIGH
                spd = SPEED_LOW if near_drop else SPEED_HIGH

                if abs(dx_px) <= dz and abs(dy_px) <= dz:
                    # CENTERED
                    m_fwd, m_right = pixels_to_meters(dx_px, dy_px, cur_alt)
                    dist_m = math.sqrt(m_fwd**2 + m_right**2)

                    log(log_f, f"CENTERED at {cur_alt:.1f}m "
                               f"({dx_px:+d},{dy_px:+d})px = {dist_m:.2f}m")
                    if not args.dry_run:
                        fc.stop()

                    if frame is not None:
                        frame_count[0] += process_frame(
                            frame, state, det, cur_alt, fc, vw,
                            centered=True,
                            mission_elapsed=time.time() - mission_t0)

                    if cur_alt <= drop_alt + 0.8:
                        state = "DROP_ALIGNMENT"
                        drop_window.reset()
                        drop_t0 = time.time()
                        log(log_f, f"→ DROP_CENTERING (alt={cur_alt:.1f}m)")
                    else:
                        state = "DESCENDING"
                        descend_tgt = max(cur_alt - DESCEND_STEP, drop_alt)
                        log(log_f, f"→ DESCENDING to {descend_tgt:.1f}m")
                    time.sleep(0.5)
                    continue

                # Not centered — correct
                m_fwd, m_right = pixels_to_meters(dx_px, dy_px, cur_alt)
                dist_m = math.sqrt(m_fwd**2 + m_right**2)

                # Skip micro-corrections that wind overwhelms at cruise alt
                if dist_m < MIN_CORRECT_DIST:
                    if not args.dry_run:
                        fc.stop()
                    csv_row(csv_f, state, fc, cur_alt, det,
                            dx_px, dy_px, m_fwd, m_right, dist_m,
                            0, 0, 0,
                            mission_elapsed=time.time() - mission_t0,
                            frame_num=frame_count[0],
                            notes=f"skip_micro_{dist_m:.2f}m")
                    print(f"\r  [ACQUIRING] offset {dist_m:.2f}m < "
                          f"{MIN_CORRECT_DIST}m — holding   ",
                          end="", flush=True)
                    time.sleep(VEL_RATE)
                    continue

                scale = min(spd / dist_m, 1.0) if dist_m > spd else 0.5
                vx = m_fwd * scale
                vy = m_right * scale

                if not args.dry_run:
                    fc.velocity_body(vx, vy, 0)

                csv_row(csv_f, state, fc, cur_alt, det,
                        dx_px, dy_px, m_fwd, m_right, dist_m, vx, vy, 0,
                        mission_elapsed=time.time() - mission_t0,
                        frame_num=frame_count[0])

                if frame is not None:
                    frame_count[0] += process_frame(
                        frame, state, det, cur_alt, fc, vw,
                        mission_elapsed=time.time() - mission_t0,
                        extra_info=f"offset={dist_m:.2f}m")

                # Patience: if centering too long at high alt, just descend
                acquire_elapsed = time.time() - acquire_t0
                if acquire_elapsed > ACQUIRE_PATIENCE and cur_alt > drop_alt + 0.8:
                    log(log_f, f"PATIENCE EXPIRED ({acquire_elapsed:.0f}s) — descending anyway")
                    if not args.dry_run:
                        fc.stop()
                    state = "DESCENDING"
                    descend_tgt = max(cur_alt - DESCEND_STEP, drop_alt)
                    time.sleep(0.5)
                    continue

                time.sleep(VEL_RATE)

            # ══════════════════════════════════════════════════
            # DESCEND
            # ══════════════════════════════════════════════════
            elif state == "DESCENDING":
                log(log_f, f"DESCEND {cur_alt:.1f}m → {descend_tgt:.1f}m")
                t0 = time.time()
                while True:
                    if not args.dry_run:
                        fc.poll()
                    cur_alt = fc.alt if not args.dry_run else descend_tgt
                    if cur_alt <= descend_tgt + 0.3:
                        break
                    if time.time() - t0 > 15:
                        break
                    if not args.dry_run:
                        fc.velocity_ned(0, 0, DESCENT_VZ)

                    if cap:
                        ret, frm = cap.read()
                        if ret:
                            d = detect_x(frm, model, args.conf, args.imgsz)
                            if d:
                                last_x = time.time()
                            frame_count[0] += process_frame(
                                frm, "DESCENDING", d, cur_alt, fc, vw,
                                mission_elapsed=time.time() - mission_t0)
                    time.sleep(VEL_RATE)

                if not args.dry_run:
                    fc.stop()
                time.sleep(1)
                acquire_t0 = time.time()
                state = "ACQUIRING"

            # ══════════════════════════════════════════════════
            # DROP CENTERING — rolling window at drop altitude
            # ══════════════════════════════════════════════════
            elif state == "DROP_ALIGNMENT":
                drop_elapsed = time.time() - drop_t0
                if drop_elapsed > DROP_TIMEOUT:
                    log(log_f, f"DROP TIMEOUT ({DROP_TIMEOUT}s) → DROPPING & RTL")
                    if not args.dry_run:
                        fc.stop()
                        safe_rtl(fc, log_f, cap, vw, frame_count, mission_t0,
                                 drop_first=True)
                    state = "ABORT"; continue

                if det is None:
                    lost = time.time() - last_x
                    if lost > LOST_TIMEOUT:
                        log(log_f, f"LOST during drop centering {lost:.0f}s → DROPPING & RTL")
                        if not args.dry_run:
                            fc.stop()
                            safe_rtl(fc, log_f, cap, vw, frame_count, mission_t0,
                                     drop_first=True)
                        state = "ABORT"; continue
                    if not args.dry_run:
                        fc.stop()
                    drop_window.add_frame(False, fc.lat, fc.lon)
                    print(f"\r  [DROP] {drop_window.summary_str()} | Lost X ({lost:.1f}s)   ",
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

                drop_window.add_frame(is_centered, fc.lat, fc.lon, dx_px, dy_px)

                if is_centered:
                    if not args.dry_run:
                        fc.stop()
                    if frame is not None:
                        di = {
                            'n_centered': drop_window.centered_count,
                            'n_total': drop_window.total_frames,
                            'required': DROP_MIN_CENTERED,
                            'hit_rate': drop_window.hit_rate,
                            'window_entries': [e[1] for e in drop_window.buffer],
                        }
                        frame_count[0] += process_frame(
                            frame, state, det, cur_alt, fc, vw,
                            centered=True, drop_info=di,
                            mission_elapsed=time.time() - mission_t0)
                else:
                    # Not centered — correct, but skip micro-corrections
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
                        mission_elapsed=time.time() - mission_t0,
                        frame_num=frame_count[0],
                        notes='HIT' if is_centered else 'MISS')

                print(f"\r  [DROP] {drop_window.summary_str()} | "
                      f"({dx_px:+d},{dy_px:+d})px   ",
                      end="", flush=True)

                # Check if ready to drop
                drop_result = drop_window.check_drop_ready()
                if drop_result:
                    drop_lat = drop_result['lat']
                    drop_lon = drop_result['lon']
                    drop_locked = True

                    log(log_f, "")
                    log(log_f, "=" * 55)
                    log(log_f, "  ★ ★ ★  DROP LOCK CONFIRMED  ★ ★ ★")
                    log(log_f, "=" * 55)
                    log(log_f, f"  Centered:   {drop_result['n_centered']}/{drop_result['n_total']} "
                               f"({drop_result['hit_rate']:.0%})")
                    log(log_f, f"  GPS spread: {drop_result['spread']:.3f}m")
                    log(log_f, f"  Drop pos:   ({drop_lat:.8f}, {drop_lon:.8f})")
                    log(log_f, f"  Altitude:   {cur_alt:.1f}m")
                    log(log_f, "=" * 55)

                    print(f"\n\n  ★ DROP LOCKED at ({drop_lat:.8f}, {drop_lon:.8f})")
                    print(f"    → SCOOT FWD {BLIND_SCOOT_DIST}m then DROP!\n")

                    state = "DROPPING"
                    if not args.dry_run:
                        fc.stop()
                    time.sleep(0.3)
                    continue

                time.sleep(VEL_RATE)

            # ══════════════════════════════════════════════════
            # DROP — BLIND SCOOT + OPEN CLAW
            # ══════════════════════════════════════════════════
            elif state == "DROPPING":
                # ── BLIND SCOOT: camera is over X, move fwd ──
                # Camera is centered on X but the claw is behind
                # the camera. Scoot forward blindly to put the
                # claw directly over X, then drop.
                scoot_time = BLIND_SCOOT_DIST / BLIND_SCOOT_SPEED

                log(log_f, "")
                log(log_f, "=" * 55)
                log(log_f, f"  → BLIND SCOOT: {BLIND_SCOOT_DIST:.2f}m fwd "
                           f"@ {BLIND_SCOOT_SPEED:.2f}m/s "
                           f"({scoot_time:.1f}s)")
                log(log_f, "=" * 55)

                if not args.dry_run:
                    scoot_t0 = time.time()
                    while time.time() - scoot_t0 < scoot_time:
                        fc.velocity_body(BLIND_SCOOT_SPEED, 0, 0)
                        fc.poll()
                        if cap:
                            ret, frm = cap.read()
                            if ret:
                                frame_count[0] += process_frame(
                                    frm, "DROPPING", None, fc.alt, fc, vw,
                                    dropped=False,
                                    mission_elapsed=time.time() - mission_t0)
                        time.sleep(VEL_RATE)
                    fc.stop()

                log(log_f, f"  Scoot done — holding {BLIND_SCOOT_HOLD}s to settle")
                if not args.dry_run:
                    time.sleep(BLIND_SCOOT_HOLD)

                csv_row(csv_f, "SCOOT_DONE", fc, cur_alt,
                        drop_lat=drop_lat, drop_lon=drop_lon,
                        mission_elapsed=time.time() - mission_t0,
                        frame_num=frame_count[0],
                        notes=f"scoot_{BLIND_SCOOT_DIST:.2f}m_fwd")

                # ── NOW OPEN THE CLAW ────────────────────────
                log(log_f, "")
                log(log_f, "=" * 55)
                log(log_f, "  📦  OPENING CLAW — PAYLOAD AWAY!")
                log(log_f, "=" * 55)

                if not args.dry_run:
                    set_claw(fc, CLAW_OPEN_PWM, log_f)
                    for _ in range(5):
                        set_claw(fc, CLAW_OPEN_PWM)
                        time.sleep(0.1)
                else:
                    log(log_f, f"DRY RUN: would send RC{CLAW_CHANNEL}={CLAW_OPEN_PWM}")

                payload_dropped = True
                csv_row(csv_f, "DROPPING", fc, cur_alt,
                        drop_lat=drop_lat, drop_lon=drop_lon,
                        mission_elapsed=time.time() - mission_t0,
                        frame_num=frame_count[0],
                        notes=f"CLAW_OPEN")

                # Hold for payload to fall
                log(log_f, f"Holding {POST_DROP_HOLD}s for payload to clear")
                hold_t0 = time.time()
                while time.time() - hold_t0 < POST_DROP_HOLD:
                    if not args.dry_run:
                        fc.poll()
                        set_claw(fc, CLAW_OPEN_PWM)
                    if cap:
                        ret, frm = cap.read()
                        if ret:
                            frame_count[0] += process_frame(
                                frm, "DROPPING", None, cur_alt, fc, vw,
                                dropped=True,
                                mission_elapsed=time.time() - mission_t0)
                    time.sleep(0.2)

                state = "RETURNING"
                continue

            # ══════════════════════════════════════════════════
            # POST DROP — close claw, RTL
            # ══════════════════════════════════════════════════
            elif state == "RETURNING":
                log(log_f, "Post-drop → safe RTL")

                if not args.dry_run:
                    safe_rtl(fc, log_f, cap, vw, frame_count, mission_t0)

                    rtl_t0 = time.time()
                    while fc.armed and (time.time() - rtl_t0 < 120):
                        fc.poll()
                        if cap:
                            ret, frm = cap.read()
                            if ret:
                                frame_count[0] += process_frame(
                                    frm, "RTL", None, fc.alt, fc, vw,
                                    dropped=True,
                                    mission_elapsed=time.time() - mission_t0)
                        time.sleep(0.2)

                # Mission complete report
                mission_total = time.time() - mission_t0
                log(log_f, "")
                log(log_f, "=" * 60)
                log(log_f, "  ★ ★ ★  MISSION COMPLETE  ★ ★ ★")
                log(log_f, "=" * 60)
                log(log_f, f"  Drop GPS:      ({drop_lat:.8f}, {drop_lon:.8f})")
                log(log_f, f"  Drop alt:      {drop_alt:.1f}m")
                log(log_f, f"  Mission time:  {mission_total:.1f}s")
                log(log_f, f"  Battery:       {fc.battery_pct}%")
                log(log_f, "=" * 60)

                print(f"\n\n{'='*60}")
                print(f"  ★ ★ ★  PACKAGE DROPPED!  ★ ★ ★")
                if drop_locked:
                    print(f"  Drop position: ({drop_lat:.8f}, {drop_lon:.8f})")
                print(f"  Mission time:  {mission_total:.1f}s")
                print(f"{'='*60}\n")
                state = "DONE"

            # ── ABORT ─────────────────────────────────────────
            elif state == "ABORT":
                log(log_f, "ABORTED — opening claw to release payload")
                if not args.dry_run:
                    set_claw(fc, CLAW_OPEN_PWM, log_f)
                    for _ in range(5):
                        set_claw(fc, CLAW_OPEN_PWM)
                        time.sleep(0.1)
                    time.sleep(POST_DROP_HOLD)
                    set_claw(fc, CLAW_CLOSE_PWM, log_f)
                    time.sleep(0.5)
                    release_rc_override(fc)
                    fc.wait_disarmed(timeout=60)
                state = "DONE"

    # ── Finalize video ────────────────────────────────────────
    if vw:
        vw.release()
        record_elapsed = time.time() - record_t0 if record_t0 else 1
        measured_fps = frame_count[0] / max(record_elapsed, 0.001)
        log(log_f, f"Video: {frame_count[0]} frames, {record_elapsed:.1f}s, "
                   f"{measured_fps:.1f} FPS")

        if frame_count[0] > 0 and os.path.isfile(video_path_tmp):
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
                log(log_f, f"ffmpeg remux failed ({e}), raw: {video_path}")
        else:
            if os.path.isfile(video_path_tmp):
                os.rename(video_path_tmp, video_path)

    csv_f.close()
    log_f.close()

    print(f"\n[*] Flight log:  {log_fname}")
    print(f"[*] CSV data:    {csv_fname}")
    if video_path:
        print(f"[*] Video:       {video_path}")
    print(f"[*] Done!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Mother Mission — Full Autonomous Delivery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 mother.py --lat 33.78310 --lon -118.10940
  python3 mother.py --lat 33.78310 --lon -118.10940 --speed 0.5 --alt 5
  python3 mother.py --lat 33.78310 --lon -118.10940 --alt 8 --max-alt 20 --drop-alt 3
  python3 mother.py --lat 33.78310 --lon -118.10940 --cross-dist 5
  python3 mother.py --lat 33.78310 --lon -118.10940 --dry-run
  python3 mother.py --lat 33.78310 --lon -118.10940 --sitl
""")

    # ── GPS target ──
    p.add_argument("--lat", type=float, required=True,
                   help="Target latitude (decimal degrees)")
    p.add_argument("--lon", type=float, required=True,
                   help="Target longitude (decimal degrees)")

    # ── Flight params ──
    p.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                   help=f"Cruise speed in m/s (default {DEFAULT_SPEED})")
    p.add_argument("--alt", type=float, default=TAKEOFF_ALT,
                   help=f"Takeoff/cruise altitude in meters (default {TAKEOFF_ALT})")
    p.add_argument("--max-alt", type=float, default=15.0,
                   help="Maximum search altitude in meters (default 15)")
    p.add_argument("--drop-alt", type=float, default=DROP_ALT_DEFAULT,
                   help=f"Altitude to hover and drop payload (default {DROP_ALT_DEFAULT})")

    # ── YOLO ──
    p.add_argument("--weights", default="gol.pt",
                   help="YOLO weights file")
    p.add_argument("--conf", type=float, default=0.60,
                   help="YOLO confidence threshold")
    p.add_argument("--imgsz", type=int, default=640,
                   help="YOLO input size")

    # ── Modes ──
    p.add_argument("--dry-run", action="store_true",
                   help="Camera only, no flight (test detection)")
    p.add_argument("--sitl", action="store_true",
                   help="SITL mode (webcam + simulated FC)")
    p.add_argument("--feed-port", type=int, default=5000,
                   help="Port for live browser feed (default 5000)")

    # ── Mission limits ──
    p.add_argument("--cross-dist", type=float, default=CROSS_PROBE_DIST,
                   help=f"Cross-sweep leg distance in meters (default {CROSS_PROBE_DIST})")

    main(p.parse_args())
