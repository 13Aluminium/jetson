#!/usr/bin/env python3
"""
recon_mission.py — Reconnaissance / CV Validation Flight
=========================================================
Stripped-down mother.py for a quick 3-minute recon run.

Purpose:
    Fly to target GPS, test YOLO model on the REAL target,
    do centering + descent (but NO drop), and capture photos
    every second for potential overnight retraining.

What it does:
    1. LAUNCH to cruise altitude (5m)
    2. TRANSIT to GPS coordinates at 2 m/s (fast — only ~30m)
    3. SEARCH for X (visual scan → rotation scan → altitude step)
    4. ACQUIRING — YOLO centering corrections over X
    5. DESCENDING — step down to recon altitude (3m)
    6. RECON HOLD — hold at 3m, keep centering, keep taking photos
       (validates the full CV pipeline at actual drop altitude)
    7. TIMER FIRES (180s) → climb to 5m → RTL

What it does NOT do:
    - No claw commands (no open, no close)
    - No payload drop
    - No blind scoot
    - No rolling window latch
    - No package delivery scoring

Photo capture:
    Background thread saves a JPEG every ~1 second with filename:
        recon_YYYYMMDD_HHMMSS/frame_NNNN_alt_X.Xm_lat_XX.XXXXXXXX_lon_XX.XXXXXXXX.jpg
    These are RAW camera frames (no overlay) for training data.
    The video recording has overlays for review.

Usage:
    python3 recon_mission.py --lat 35.05XXX --lon -118.15XXX

    python3 recon_mission.py --lat 35.05XXX --lon -118.15XXX --speed 2.0 \\
                             --alt 5 --recon-alt 3

    python3 recon_mission.py --lat 35.05XXX --lon -118.15XXX --dry-run

Terminal 1: mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 \\
            --out=udp:127.0.0.1:14551
Terminal 2: python3 recon_mission.py --lat <lat> --lon <lon>

Failsafes:
    Ctrl+C → RTL | Exception → RTL | 3-min timer → climb 5m + RTL
    X lost 10s during centering → RTL | Battery < 20% → RTL
"""

import argparse
import math
import os
import sys
import time
import cv2
import threading
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
DEFAULT_SPEED     = 2.0     # m/s — FAST transit (only 30m)
ARRIVE_RADIUS     = 1.5     # meters — "arrived" at target GPS
NAV_TIMEOUT       = 60      # seconds max for GPS navigation

# ── Search pattern ────────────────────────────────────────────
YAW_STEP_DEG      = 90
YAW_SPEED         = 20      # °/s
YAW_SETTLE        = 1.0
YAW_SCAN_DWELL    = 2.0
SEARCH_DWELL      = 3.0
ALT_CLIMB_STEP    = 2.0
CROSS_PROBE_DIST  = 3.0
CROSS_PROBE_SPEED = 0.5
HEADING_TOL       = 5.0

# ── YOLO centering ───────────────────────────────────────────
DESCEND_STEP      = 1.0
DEADZONE_HIGH     = 60      # px at cruise alt
DEADZONE_DROP     = 80      # px at recon alt
SPEED_HIGH        = 0.30    # m/s centering speed at cruise
SPEED_LOW         = 0.15    # m/s centering speed near recon alt
RECON_ALT_DEFAULT = 3.0     # meters — hover and observe

# ── Safety ────────────────────────────────────────────────────
MISSION_TIME_LIMIT_DEFAULT = 180    # 3 MINUTES — hard RTL
MISSION_TIME_LIMIT = MISSION_TIME_LIMIT_DEFAULT  # overridden by --timer
LOST_TIMEOUT       = 10.0   # seconds without YOLO → RTL
VEL_RATE           = 0.2
DESCENT_VZ         = 0.30
ACQUIRE_PATIENCE   = 15.0
BATTERY_MIN        = 20
SAFE_RTL_ALT       = 5.0    # meters — climb to this before RTL
MIN_CORRECT_DIST   = 0.5
MIN_CORRECT_DIST_RECON = 0.2

# ── Photo capture ────────────────────────────────────────────
PHOTO_INTERVAL     = 1.0    # seconds between photo saves

# ── Video / overlay ──────────────────────────────────────────
OVERLAY_FONT      = cv2.FONT_HERSHEY_SIMPLEX
COLOR_OK          = (0, 255, 0)
COLOR_LOST        = (0, 0, 255)
COLOR_CENTER      = (0, 255, 255)
COLOR_RECON       = (255, 165, 0)    # orange for recon hold
COLOR_NAV         = (255, 200, 0)
FEED_QUALITY      = 60


# ===========================================================================
# LIVE FEED (Flask MJPEG server)
# ===========================================================================
latest_frame = None
frame_lock = threading.Lock()
flask_app = Flask(__name__)

FEED_HTML = """<!DOCTYPE html><html><head><title>Recon Mission</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{background:#111;color:#eee;
font-family:system-ui;display:flex;flex-direction:column;align-items:center;
min-height:100vh}h1{margin:15px 0 5px;font-size:1.3em;color:#ff8c00}
.info{font-size:.8em;color:#666;margin-bottom:10px}.info span{color:#ff8c00}
img{max-width:95vw;max-height:84vh;border:2px solid #333;border-radius:6px}
</style></head><body><h1>RECON Mission — Live Feed</h1>
<p class="info">Mode: <span>CV VALIDATION (NO DROP)</span></p>
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
    print(f"  RECON LIVE FEED on http://{ip}:{port}")
    print(f"{'='*55}\n")
    t = threading.Thread(
        target=lambda: flask_app.run(host='0.0.0.0', port=port, threaded=True,
                                     use_reloader=False),
        daemon=True)
    t.start()
    return t


# ===========================================================================
# GPS HELPERS (from mother.py)
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


def bearing_to_compass(deg):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[round(deg / 22.5) % 16]


# ===========================================================================
# SAFE RTL (no claw logic — just climb and go home)
# ===========================================================================

def safe_rtl(fc, log_f, cap=None, vw=None, frame_count_ref=None,
             mission_t0=None, safe_alt=SAFE_RTL_ALT):
    """Climb to safe altitude before commanding RTL."""
    fc.poll()
    cur = fc.alt
    if cur < safe_alt - 0.5:
        log(log_f, f"SAFE RTL: climbing {cur:.1f}m -> {safe_alt:.1f}m before RTL")
        t0 = time.time()
        while time.time() - t0 < 20:
            fc.poll()
            if fc.alt >= safe_alt - 0.3:
                break
            fc.velocity_ned(0, 0, -0.5)
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
# NAVIGATION HELPERS (from mother.py)
# ===========================================================================

def send_goto(fc, lat, lon, alt):
    fc.master.mav.set_position_target_global_int_send(
        0, fc.master.target_system, fc.master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0b0000_1111_1111_1000,
        int(lat * 1e7), int(lon * 1e7), alt,
        0, 0, 0, 0, 0, 0, 0, 0)


def send_yaw(fc, angle_deg, speed_deg_s, direction, relative=True):
    fc.master.mav.command_long_send(
        fc.master.target_system, fc.master.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0,
        abs(angle_deg), speed_deg_s, direction,
        1 if relative else 0, 0, 0, 0)


def set_speed(fc, speed_mps):
    fc.master.mav.command_long_send(
        fc.master.target_system, fc.master.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 0,
        0, speed_mps, -1, 0, 0, 0, 0)


def normalize_heading(h):
    return h % 360


def wait_yaw_complete(fc, target_heading, timeout=30):
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
    dist = math.sqrt(fwd**2 + right**2)
    if dist < 0.01:
        return None
    travel_time = dist / speed
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
    time.sleep(0.5)
    return None


# ===========================================================================
# VIDEO OVERLAY
# ===========================================================================

def draw_overlay(frame, state, det, cur_alt, fc,
                 centered=False, mission_elapsed=0, extra_info="",
                 recon_stats=None):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Crosshair
    cv2.line(frame, (cx - 30, cy), (cx + 30, cy), COLOR_CENTER, 1)
    cv2.line(frame, (cx, cy - 30), (cx, cy + 30), COLOR_CENTER, 1)

    # Deadzone circle
    if state == "RECON_HOLD":
        cv2.circle(frame, (cx, cy), DEADZONE_DROP, COLOR_RECON, 2)
        cv2.circle(frame, (cx, cy), DEADZONE_DROP // 2, COLOR_RECON, 1)
    elif cur_alt < RECON_ALT_DEFAULT + 1.5:
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
    elif state not in ("LAUNCH", "TRANSIT"):
        cv2.putText(frame, "NO X", (cx - 30, cy + 50),
                    OVERLAY_FONT, 0.8, COLOR_LOST, 2)

    # RECON MODE banner
    cv2.putText(frame, "RECON - NO DROP", (w - 280, h - 15),
                OVERLAY_FONT, 0.7, COLOR_RECON, 2)

    # Phase indicator
    phase_colors = {
        "LAUNCH": COLOR_OK, "TRANSIT": COLOR_NAV,
        "VISUAL_SCAN": COLOR_CENTER, "ROTATION_SCAN": COLOR_CENTER,
        "ALTITUDE_STEP": COLOR_CENTER, "CROSS_SWEEP": COLOR_CENTER,
        "ACQUIRING": COLOR_OK, "DESCENDING": COLOR_OK,
        "RECON_HOLD": COLOR_RECON,
    }
    pc = phase_colors.get(state, COLOR_OK)
    cv2.putText(frame, f"PHASE: {state}", (w - 320, 25),
                OVERLAY_FONT, 0.6, pc, 2)

    # Mission timer (countdown from 180s)
    remaining = max(0, MISSION_TIME_LIMIT - mission_elapsed)
    timer_color = COLOR_LOST if remaining < 30 else COLOR_OK
    cv2.putText(frame, f"T-{remaining:.0f}s", (w - 120, 55),
                OVERLAY_FONT, 0.6, timer_color, 2)

    # Recon stats bar
    if recon_stats:
        stats_text = (f"Photos: {recon_stats['photos']} | "
                      f"Detections: {recon_stats['detections']} | "
                      f"Centered: {recon_stats['centered']}")
        cv2.putText(frame, stats_text, (10, h - 45),
                    OVERLAY_FONT, 0.55, COLOR_RECON, 2)

    # HUD
    hud_color = COLOR_OK if det else COLOR_LOST
    lines = [
        f"STATE: {state}",
        f"ALT: {cur_alt:.1f}m",
        f"GPS: {fc.lat:.7f}, {fc.lon:.7f}",
        f"SATS: {fc.satellites}  FIX: {fc.gps_fix}",
        f"BATT: {fc.battery_pct}%",
        f"HDG: {fc.heading:.0f}deg",
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
                  centered=False, mission_elapsed=0, extra_info="",
                  recon_stats=None):
    if frame is None:
        return 0
    overlay = draw_overlay(frame.copy(), state, det, cur_alt, fc,
                           centered=centered,
                           mission_elapsed=mission_elapsed,
                           extra_info=extra_info,
                           recon_stats=recon_stats)
    if vw:
        vw.write(overlay)
    update_feed_frame(overlay)
    return 1


# ===========================================================================
# PHOTO CAPTURE THREAD
# ===========================================================================

class PhotoCapture:
    """Background thread that saves raw camera frames for training data."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.frame_queue = deque(maxlen=5)  # buffer up to 5
        self.count = 0
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()

    def capture(self, frame, alt, lat, lon, state, det_conf=0.0):
        """Queue a frame for saving. Non-blocking."""
        if frame is None:
            return
        with self.lock:
            self.frame_queue.append((
                frame.copy(), alt, lat, lon, state, det_conf,
                datetime.now().strftime("%H%M%S_%f")
            ))

    def _writer_loop(self):
        while self.running:
            item = None
            with self.lock:
                if self.frame_queue:
                    item = self.frame_queue.popleft()
            if item is None:
                time.sleep(0.05)
                continue

            frame, alt, lat, lon, state, det_conf, ts = item
            self.count += 1
            fname = (f"frame_{self.count:04d}_alt_{alt:.1f}m_"
                     f"lat_{lat:.8f}_lon_{lon:.8f}_"
                     f"conf_{det_conf:.2f}_{state}.jpg")
            path = os.path.join(self.output_dir, fname)
            try:
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            except Exception as e:
                print(f"[!] Photo save failed: {e}")

    def stop(self):
        self.running = False
        self.thread.join(timeout=3)


# ===========================================================================
# CSV DATA LOGGER
# ===========================================================================

def create_csv_log(prefix="recon_data"):
    fname = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    f = open(fname, 'w')
    f.write("timestamp,state,alt_m,lat,lon,gps_fix,satellites,"
            "heading_deg,battery_pct,"
            "yolo_detected,yolo_conf,yolo_cx,yolo_cy,"
            "dx_px,dy_px,fwd_m,right_m,dist_m,"
            "vx_cmd,vy_cmd,vz_cmd,"
            "mission_elapsed,frame_num,photo_count,notes\n")
    return fname, f


def csv_row(f, state, fc, cur_alt, det=None,
            dx_px=0, dy_px=0, fwd_m=0, right_m=0, dist_m=0,
            vx=0, vy=0, vz=0,
            mission_elapsed=0, frame_num=0, photo_count=0, notes=""):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    f.write(f"{ts},{state},{cur_alt:.2f},"
            f"{fc.lat:.8f},{fc.lon:.8f},{fc.gps_fix},{fc.satellites},"
            f"{fc.heading:.1f},{fc.battery_pct},"
            f"{1 if det else 0},{det['conf'] if det else 0:.3f},"
            f"{det['cx'] if det else 0},{det['cy'] if det else 0},"
            f"{dx_px},{dy_px},{fwd_m:.4f},{right_m:.4f},{dist_m:.4f},"
            f"{vx:.4f},{vy:.4f},{vz:.4f},"
            f"{mission_elapsed:.1f},{frame_num},{photo_count},{notes}\n")
    f.flush()


# ===========================================================================
# MAIN
# ===========================================================================

def main(args):
    global MISSION_TIME_LIMIT
    MISSION_TIME_LIMIT = args.timer

    recon_alt = args.recon_alt
    target_lat = args.lat
    target_lon = args.lon

    # ── Sanity checks ─────────────────────────────────────────
    if args.alt < recon_alt:
        print(f"[!] ERROR: Cruise alt ({args.alt}m) must be >= recon alt ({recon_alt}m)")
        sys.exit(1)
    if args.max_alt < args.alt:
        print(f"[!] ERROR: Max alt ({args.max_alt}m) must be >= cruise alt ({args.alt}m)")
        sys.exit(1)

    # ── Confirmation ──────────────────────────────────────────
    if not args.dry_run and not args.sitl:
        desc = (
            f"RECON MISSION — CV VALIDATION (NO DROP)\n"
            f"  Target GPS:  ({target_lat:.8f}, {target_lon:.8f})\n"
            f"  Cruise alt:  {args.alt}m  |  Max alt: {args.max_alt}m  |  Recon alt: {recon_alt}m\n"
            f"  Speed:       {args.speed} m/s\n"
            f"  Time limit:  {MISSION_TIME_LIMIT}s ({MISSION_TIME_LIMIT/60:.1f} min)\n"
            f"  Plan: Takeoff -> Fly to GPS -> Search -> Center -> Descend to {recon_alt}m -> HOLD (no drop) -> RTL\n"
            f"  Photos saved every {PHOTO_INTERVAL}s for training data"
        )
        if not confirm("recon_mission.py — RECON FLIGHT", desc):
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
    frame_count = [0]
    record_t0 = None

    if cap:
        actual_fps = get_camera_fps(cap, sitl=args.sitl)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path_tmp = f"recon_mission_{ts_str}_tmp.mp4"
        video_path = f"recon_mission_{ts_str}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(video_path_tmp, fourcc, actual_fps, (FRAME_W, FRAME_H))
        if not vw.isOpened():
            print("[!] WARNING: Could not open video writer")
            vw = None
        else:
            print(f"[REC] Recording -> {video_path} ({actual_fps:.1f} FPS)")

    # ── Photo capture thread ─────────────────────────────────
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    photo_dir = f"recon_photos_{ts_str}"
    photo_cap = PhotoCapture(photo_dir)
    last_photo_time = 0
    print(f"[PHOTO] Saving training photos to {photo_dir}/")

    # ── Logs ──────────────────────────────────────────────────
    log_fname, log_f = create_log("recon_mission")
    csv_fname, csv_f = create_csv_log("recon_data")

    log(log_f, "=" * 60)
    log(log_f, "RECON MISSION — CV VALIDATION (NO DROP)")
    log(log_f, "=" * 60)
    log(log_f, f"Target GPS:      ({target_lat:.8f}, {target_lon:.8f})")
    log(log_f, f"Cruise alt:      {args.alt}m")
    log(log_f, f"Max alt:         {args.max_alt}m")
    log(log_f, f"Recon alt:       {recon_alt}m")
    log(log_f, f"Speed:           {args.speed} m/s")
    log(log_f, f"Time limit:      {MISSION_TIME_LIMIT}s ({MISSION_TIME_LIMIT/60:.1f} min)")
    log(log_f, f"Photo interval:  {PHOTO_INTERVAL}s")
    log(log_f, f"Photo dir:       {photo_dir}/")
    log(log_f, f"YOLO weights:    {args.weights}")
    log(log_f, f"YOLO conf:       {args.conf}")
    log(log_f, "")

    # ── Recon stats ──────────────────────────────────────────
    recon_stats = {'photos': 0, 'detections': 0, 'centered': 0}

    with SafeFlight(fc, camera=cap, video_writer=vw) as sf:

        # ── State variables ───────────────────────────────────
        state = "LAUNCH"
        mission_t0 = time.time()
        last_x = 0
        acquire_t0 = 0
        descend_tgt = 0
        yaw_scan_step = 0
        search_alt = args.alt
        cross_phase = 0

        # Navigation
        initial_dist = 0
        travel_heading = 0

        if args.dry_run:
            state = "VISUAL_SCAN"
            log(log_f, "DRY RUN — skipping takeoff and navigation")

        # ══════════════════════════════════════════════════════
        # STATE MACHINE
        # ══════════════════════════════════════════════════════
        while state not in ("DONE", "ABORT"):

            # ── Global mission timer ─────────────────────────
            mission_elapsed = time.time() - mission_t0
            if mission_elapsed > MISSION_TIME_LIMIT and state not in (
                    "DONE", "ABORT"):
                log(log_f, f"MISSION TIME LIMIT ({MISSION_TIME_LIMIT}s) -> climb to {SAFE_RTL_ALT}m + RTL")
                csv_row(csv_f, state, fc, fc.alt if not args.dry_run else args.alt,
                        mission_elapsed=mission_elapsed,
                        frame_num=frame_count[0],
                        photo_count=photo_cap.count,
                        notes="TIME_LIMIT_RTL")
                if not args.dry_run:
                    fc.stop()
                    safe_rtl(fc, log_f, cap, vw, frame_count, mission_t0)
                state = "ABORT"
                continue

            # ── Battery check ────────────────────────────────
            if not args.dry_run:
                fc.poll()
                if 0 < fc.battery_pct < BATTERY_MIN and state not in (
                        "DONE", "ABORT"):
                    log(log_f, f"BATTERY LOW ({fc.battery_pct}%) -> RTL")
                    fc.stop()
                    safe_rtl(fc, log_f, cap, vw, frame_count, mission_t0)
                    state = "ABORT"
                    continue

            cur_alt = fc.alt if (not args.dry_run and fc.alt > 0.3) else args.alt

            # ── Read camera ──────────────────────────────────
            det = None
            frame = None
            if cap and state not in ("LAUNCH", "TRANSIT"):
                ret, frame = cap.read()
                if ret:
                    det = detect_x(frame, model, args.conf, args.imgsz)
                else:
                    frame = None

            # ── Update recon stats ───────────────────────────
            if det:
                recon_stats['detections'] += 1

            # ── Background photo capture (every PHOTO_INTERVAL) ──
            now = time.time()
            if frame is not None and (now - last_photo_time) >= PHOTO_INTERVAL:
                photo_cap.capture(frame, cur_alt, fc.lat, fc.lon,
                                  state, det['conf'] if det else 0.0)
                recon_stats['photos'] = photo_cap.count
                last_photo_time = now

            # ── Process frame for video/feed ─────────────────
            if frame is not None and state not in ("LAUNCH", "TRANSIT"):
                frame_count[0] += process_frame(
                    frame, state, det, cur_alt, fc, vw,
                    mission_elapsed=time.time() - mission_t0,
                    recon_stats=recon_stats)

            # ══════════════════════════════════════════════════
            # LAUNCH
            # ══════════════════════════════════════════════════
            if state == "LAUNCH":
                log(log_f, f"LAUNCH -> {args.alt}m")
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

                log(log_f, f"At {fc.alt:.1f}m — stabilizing 3s")
                t0 = time.time()
                while time.time() - t0 < 3:
                    if cap:
                        ret, frm = cap.read()
                        if ret:
                            fc.poll()
                            frame_count[0] += process_frame(
                                frm, "LAUNCH", None, fc.alt, fc, vw,
                                mission_elapsed=time.time() - mission_t0)
                    time.sleep(0.05)

                initial_dist = haversine(fc.lat, fc.lon, target_lat, target_lon)
                travel_heading = bearing(fc.lat, fc.lon, target_lat, target_lon)
                compass = bearing_to_compass(travel_heading)

                log(log_f, f"GPS: ({fc.lat:.8f}, {fc.lon:.8f}) sats={fc.satellites}")
                log(log_f, f"Target: ({target_lat:.8f}, {target_lon:.8f})")
                log(log_f, f"Distance: {initial_dist:.1f}m  Bearing: {travel_heading:.1f} ({compass})")

                set_speed(fc, args.speed)
                log(log_f, f"Speed set to {args.speed} m/s")

                state = "TRANSIT"
                continue

            # ══════════════════════════════════════════════════
            # TRANSIT — fly to target GPS
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

                    if mission_elapsed > MISSION_TIME_LIMIT:
                        log(log_f, "TIME LIMIT during navigation -> RTL")
                        fc.stop()
                        safe_rtl(fc, log_f, cap, vw, frame_count, mission_t0)
                        state = "ABORT"; break

                    if now - last_cmd >= 2.0:
                        send_goto(fc, target_lat, target_lon, args.alt)
                        last_cmd = now

                    remaining = haversine(fc.lat, fc.lon, target_lat, target_lon)
                    pct = max(0, (1 - remaining / initial_dist)) * 100 if initial_dist > 0 else 100

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
                            photo_count=photo_cap.count,
                            notes=f"remaining={remaining:.1f}m")

                    print(f"\r  [NAV] {pct:5.1f}%  rem={remaining:.1f}m  "
                          f"alt={fc.alt:.1f}m  sats={fc.satellites}  "
                          f"batt={fc.battery_pct}%  T-{MISSION_TIME_LIMIT-mission_elapsed:.0f}s   ",
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
                fc.poll()
                travel_heading = fc.heading
                log(log_f, f"Arrival heading: {travel_heading:.1f}")

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
                log(log_f, f"VISUAL SCAN — looking ahead (heading {fc.heading:.0f}) "
                           f"for {SEARCH_DWELL}s")

                found = scan_for_x(cap, model, args.conf, args.imgsz,
                                   SEARCH_DWELL, fc, vw, "VISUAL_SCAN",
                                   mission_t0, frame_count, log_f)
                if found:
                    det = found
                    last_x = time.time()
                    acquire_t0 = time.time()
                    state = "ACQUIRING"
                    log(log_f, f"X found in visual scan! -> ACQUIRING")
                    continue

                log(log_f, "X not found straight ahead -> ROTATION SCAN")
                yaw_scan_step = 0
                state = "ROTATION_SCAN"
                continue

            # ══════════════════════════════════════════════════
            # ROTATION SCAN — 4 x 90 CCW
            # ══════════════════════════════════════════════════
            elif state == "ROTATION_SCAN":
                if yaw_scan_step >= 4:
                    log(log_f, f"360 scan complete at alt={cur_alt:.1f}m — no X")

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

                step_num = yaw_scan_step + 1
                log(log_f, f"ROTATION SCAN step {step_num}/4 — rotating 90 CCW")

                target_hdg = normalize_heading(fc.heading - YAW_STEP_DEG)
                if not args.dry_run:
                    send_yaw(fc, YAW_STEP_DEG, YAW_SPEED, -1, relative=True)
                    timeout = max(YAW_STEP_DEG / YAW_SPEED * 2, 15)
                    wait_yaw_complete(fc, target_hdg, timeout=timeout)

                time.sleep(YAW_SETTLE)

                log(log_f, f"Scanning at heading {fc.heading:.0f} for {YAW_SCAN_DWELL}s")
                found = scan_for_x(cap, model, args.conf, args.imgsz,
                                   YAW_SCAN_DWELL, fc, vw, "ROTATION_SCAN",
                                   mission_t0, frame_count, log_f)
                if found:
                    det = found
                    last_x = time.time()
                    acquire_t0 = time.time()
                    state = "ACQUIRING"
                    log(log_f, f"X found during rotation scan step {step_num}! -> ACQUIRING")
                    continue

                yaw_scan_step += 1
                csv_row(csv_f, state, fc, cur_alt,
                        mission_elapsed=time.time() - mission_t0,
                        frame_num=frame_count[0],
                        photo_count=photo_cap.count,
                        notes=f"yaw_step_{step_num}_no_x")
                continue

            # ══════════════════════════════════════════════════
            # ALTITUDE STEP — climb for another 360 scan
            # ══════════════════════════════════════════════════
            elif state == "ALTITUDE_STEP":
                log(log_f, f"CLIMBING {cur_alt:.1f}m -> {search_alt:.1f}m")

                if not args.dry_run:
                    t0 = time.time()
                    while True:
                        fc.poll()
                        if fc.alt >= search_alt - 0.5:
                            break
                        if time.time() - t0 > 20:
                            log(log_f, "Climb timeout 20s")
                            break
                        fc.velocity_ned(0, 0, -0.5)

                        if cap:
                            ret, frm = cap.read()
                            if ret:
                                d = detect_x(frm, model, args.conf, args.imgsz)
                                frame_count[0] += process_frame(
                                    frm, "ALTITUDE_STEP", d, fc.alt, fc, vw,
                                    mission_elapsed=time.time() - mission_t0)
                                if d:
                                    fc.stop()
                                    det = d
                                    last_x = time.time()
                                    acquire_t0 = time.time()
                                    state = "ACQUIRING"
                                    log(log_f, f"X found while climbing! -> ACQUIRING")
                                    break
                        time.sleep(VEL_RATE)

                    if state == "ACQUIRING":
                        continue
                    fc.stop()

                log(log_f, f"At {fc.alt:.1f}m — starting new 360 scan")
                time.sleep(1)
                yaw_scan_step = 0
                state = "ROTATION_SCAN"
                continue

            # ══════════════════════════════════════════════════
            # CROSS SWEEP — probe fwd/back/left/right
            # ══════════════════════════════════════════════════
            elif state == "CROSS_SWEEP":
                cross_legs = [
                    (0,    0,  3.0, "FWD 3m"),
                    (180,  1,  6.0, "FLIP+FWD 6m"),
                    (180,  1,  3.0, "FLIP+FWD 3m back"),
                    (90,  -1,  3.0, "CCW 90+FWD 3m"),
                    (180,  1,  6.0, "FLIP+FWD 6m"),
                    (180,  1,  3.0, "FLIP+FWD 3m back"),
                ]

                if cross_phase >= len(cross_legs):
                    log(log_f, "CROSS SWEEP EXHAUSTED — X not found -> RTL")
                    csv_row(csv_f, state, fc, cur_alt,
                            mission_elapsed=time.time() - mission_t0,
                            frame_num=frame_count[0],
                            photo_count=photo_cap.count,
                            notes="CROSS_EXHAUSTED_RTL")
                    if not args.dry_run:
                        fc.stop()
                        safe_rtl(fc, log_f, cap, vw, frame_count, mission_t0)
                    state = "ABORT"
                    continue

                yaw_angle, yaw_dir, move_dist, label = cross_legs[cross_phase]
                leg_num = cross_phase + 1
                log(log_f, f"CROSS SWEEP leg {leg_num}/6: {label}")

                if yaw_angle > 0 and not args.dry_run:
                    target_hdg = normalize_heading(
                        fc.heading + (yaw_angle * yaw_dir if yaw_dir != 0 else 0))
                    send_yaw(fc, yaw_angle, YAW_SPEED, yaw_dir, relative=True)
                    yaw_timeout = max(yaw_angle / YAW_SPEED * 2, 10)
                    wait_yaw_complete(fc, target_hdg, timeout=yaw_timeout)
                    time.sleep(YAW_SETTLE)
                    log(log_f, f"  Yaw complete — heading {fc.heading:.0f}")

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
                        log(log_f, f"X found during cross sweep leg {leg_num}! -> ACQUIRING")
                        continue

                cross_phase += 1
                csv_row(csv_f, state, fc, cur_alt,
                        mission_elapsed=time.time() - mission_t0,
                        frame_num=frame_count[0],
                        photo_count=photo_cap.count,
                        notes=f"cross_leg_{leg_num}_no_x")
                continue

            # ══════════════════════════════════════════════════
            # ACQUIRING — YOLO centering over X
            # ══════════════════════════════════════════════════
            elif state == "ACQUIRING":
                if det is None:
                    lost = time.time() - last_x
                    if lost > LOST_TIMEOUT:
                        # RECON BEHAVIOR: descend instead of RTL
                        # Target gets bigger in frame as we go lower,
                        # and we want photos at every altitude regardless
                        if not args.dry_run:
                            fc.stop()

                        if cur_alt > recon_alt + 0.8:
                            # Still above recon alt — descend to get closer
                            log(log_f, f"LOST X {lost:.0f}s at {cur_alt:.1f}m — "
                                       f"DESCENDING anyway (recon mode, target may reappear)")
                            csv_row(csv_f, state, fc, cur_alt,
                                    mission_elapsed=time.time() - mission_t0,
                                    frame_num=frame_count[0],
                                    photo_count=photo_cap.count,
                                    notes=f"LOST_DESCEND_alt={cur_alt:.1f}")
                            state = "DESCENDING"
                            descend_tgt = max(cur_alt - DESCEND_STEP, recon_alt)
                            last_x = time.time()  # reset lost timer for next altitude
                            time.sleep(0.5)
                            continue
                        else:
                            # Already at recon alt — go to RECON_HOLD and keep taking photos
                            log(log_f, f"LOST X {lost:.0f}s at recon alt — "
                                       f"RECON_HOLD anyway (capturing photos)")
                            csv_row(csv_f, state, fc, cur_alt,
                                    mission_elapsed=time.time() - mission_t0,
                                    frame_num=frame_count[0],
                                    photo_count=photo_cap.count,
                                    notes="LOST_RECON_HOLD")
                            state = "RECON_HOLD"
                            last_x = time.time()
                            continue
                    if not args.dry_run:
                        fc.stop()
                    print(f"\r  [ACQUIRE] Lost X — holding ({lost:.1f}s / "
                          f"{LOST_TIMEOUT:.0f}s) will descend   ", end="", flush=True)
                    time.sleep(VEL_RATE)
                    continue

                last_x = time.time()
                dx_px = det['cx'] - FRAME_W // 2
                dy_px = det['cy'] - FRAME_H // 2

                near_recon = cur_alt < recon_alt + 1.5
                dz = DEADZONE_DROP if near_recon else DEADZONE_HIGH
                spd = SPEED_LOW if near_recon else SPEED_HIGH

                if abs(dx_px) <= dz and abs(dy_px) <= dz:
                    m_fwd, m_right = pixels_to_meters(dx_px, dy_px, cur_alt)
                    dist_m = math.sqrt(m_fwd**2 + m_right**2)

                    recon_stats['centered'] += 1
                    log(log_f, f"CENTERED at {cur_alt:.1f}m "
                               f"({dx_px:+d},{dy_px:+d})px = {dist_m:.2f}m")
                    if not args.dry_run:
                        fc.stop()

                    if cur_alt <= recon_alt + 0.8:
                        # At recon altitude — enter RECON_HOLD
                        state = "RECON_HOLD"
                        log(log_f, f"-> RECON_HOLD at {cur_alt:.1f}m (just observing, NO drop)")
                    else:
                        state = "DESCENDING"
                        descend_tgt = max(cur_alt - DESCEND_STEP, recon_alt)
                        log(log_f, f"-> DESCENDING to {descend_tgt:.1f}m")
                    time.sleep(0.5)
                    continue

                # Not centered — correct
                m_fwd, m_right = pixels_to_meters(dx_px, dy_px, cur_alt)
                dist_m = math.sqrt(m_fwd**2 + m_right**2)

                if dist_m < MIN_CORRECT_DIST:
                    if not args.dry_run:
                        fc.stop()
                    csv_row(csv_f, state, fc, cur_alt, det,
                            dx_px, dy_px, m_fwd, m_right, dist_m,
                            0, 0, 0,
                            mission_elapsed=time.time() - mission_t0,
                            frame_num=frame_count[0],
                            photo_count=photo_cap.count,
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
                        frame_num=frame_count[0],
                        photo_count=photo_cap.count)

                # Patience: if centering too long at high alt, just descend
                acquire_elapsed = time.time() - acquire_t0
                if acquire_elapsed > ACQUIRE_PATIENCE and cur_alt > recon_alt + 0.8:
                    log(log_f, f"PATIENCE EXPIRED ({acquire_elapsed:.0f}s) — descending anyway")
                    if not args.dry_run:
                        fc.stop()
                    state = "DESCENDING"
                    descend_tgt = max(cur_alt - DESCEND_STEP, recon_alt)
                    time.sleep(0.5)
                    continue

                print(f"\r  [ACQUIRING] offset={dist_m:.2f}m  "
                      f"({dx_px:+d},{dy_px:+d})px  "
                      f"T-{MISSION_TIME_LIMIT - mission_elapsed:.0f}s   ",
                      end="", flush=True)
                time.sleep(VEL_RATE)

            # ══════════════════════════════════════════════════
            # DESCEND
            # ══════════════════════════════════════════════════
            elif state == "DESCENDING":
                log(log_f, f"DESCEND {cur_alt:.1f}m -> {descend_tgt:.1f}m")
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
            # RECON HOLD — at recon altitude, keep centering + photos
            # ══════════════════════════════════════════════════
            elif state == "RECON_HOLD":
                # This is where the mother script would do DROP_ALIGNMENT.
                # Instead, we just keep centering and collecting data.
                # The global timer will pull us out when it expires.
                # NO RTL on lost X — just hold position and keep photographing.

                if det is None:
                    lost = time.time() - last_x
                    if not args.dry_run:
                        fc.stop()
                    remaining_t = max(0, MISSION_TIME_LIMIT - mission_elapsed)
                    print(f"\r  [RECON] Lost X ({lost:.1f}s) — holding + photographing  "
                          f"photos={photo_cap.count}  T-{remaining_t:.0f}s   ",
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

                if is_centered:
                    recon_stats['centered'] += 1
                    if not args.dry_run:
                        fc.stop()
                else:
                    # Correct position
                    if dist_m >= MIN_CORRECT_DIST_RECON:
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
                        mission_elapsed=time.time() - mission_t0,
                        frame_num=frame_count[0],
                        photo_count=photo_cap.count,
                        notes='CENTERED' if is_centered else 'CORRECTING')

                remaining_t = max(0, MISSION_TIME_LIMIT - mission_elapsed)
                status = "CENTERED" if is_centered else f"off={dist_m:.2f}m"
                print(f"\r  [RECON] {status}  "
                      f"conf={det['conf']:.0%}  "
                      f"photos={photo_cap.count}  "
                      f"T-{remaining_t:.0f}s   ",
                      end="", flush=True)

                time.sleep(VEL_RATE)

            # ── ABORT (waiting for disarm after RTL) ─────────
            elif state == "ABORT":
                if not args.dry_run:
                    fc.wait_disarmed(timeout=60)
                state = "DONE"

    # ── Stop photo capture ────────────────────────────────────
    photo_cap.stop()

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
                log(log_f, f"Remuxed -> {video_path} @ {measured_fps:.1f} FPS")
            except Exception as e:
                os.rename(video_path_tmp, video_path)
                log(log_f, f"ffmpeg remux failed ({e}), raw: {video_path}")
        else:
            if os.path.isfile(video_path_tmp):
                os.rename(video_path_tmp, video_path)

    # ── Mission summary ──────────────────────────────────────
    mission_total = time.time() - mission_t0
    log(log_f, "")
    log(log_f, "=" * 60)
    log(log_f, "  RECON MISSION COMPLETE")
    log(log_f, "=" * 60)
    log(log_f, f"  Mission time:   {mission_total:.1f}s")
    log(log_f, f"  Photos saved:   {photo_cap.count}")
    log(log_f, f"  Photo dir:      {photo_dir}/")
    log(log_f, f"  YOLO detections: {recon_stats['detections']}")
    log(log_f, f"  Centered count: {recon_stats['centered']}")
    log(log_f, f"  Battery:        {fc.battery_pct}%")
    log(log_f, "=" * 60)
    log(log_f, "")
    if recon_stats['detections'] > 0:
        log(log_f, "  CV MODEL WORKED on real target!")
        log(log_f, "  Photos can still be used to fine-tune overnight.")
    else:
        log(log_f, "  CV MODEL DID NOT DETECT TARGET")
        log(log_f, f"  -> Use {photo_cap.count} photos in {photo_dir}/ to retrain overnight")
    log(log_f, "")

    csv_f.close()
    log_f.close()

    print(f"\n\n{'='*60}")
    print(f"  RECON MISSION COMPLETE")
    print(f"  Photos: {photo_cap.count} saved to {photo_dir}/")
    print(f"  Detections: {recon_stats['detections']}")
    print(f"  Centered: {recon_stats['centered']}")
    print(f"  Time: {mission_total:.1f}s")
    print(f"{'='*60}")
    print(f"\n[*] Flight log:  {log_fname}")
    print(f"[*] CSV data:    {csv_fname}")
    if video_path:
        print(f"[*] Video:       {video_path}")
    print(f"[*] Photos:      {photo_dir}/ ({photo_cap.count} frames)")
    print(f"[*] Done!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Recon Mission — CV Validation (NO DROP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
RECON MISSION — validates YOLO model on real target, captures training photos.
NO payload drop. 3-minute hard timer.

Examples:
  python3 recon_mission.py --lat 35.05XXX --lon -118.15XXX
  python3 recon_mission.py --lat 35.05XXX --lon -118.15XXX --speed 2.0
  python3 recon_mission.py --lat 35.05XXX --lon -118.15XXX --recon-alt 3
  python3 recon_mission.py --lat 35.05XXX --lon -118.15XXX --dry-run
  python3 recon_mission.py --lat 35.05XXX --lon -118.15XXX --sitl
""")

    p.add_argument("--lat", type=float, required=True,
                   help="Target latitude (decimal degrees)")
    p.add_argument("--lon", type=float, required=True,
                   help="Target longitude (decimal degrees)")

    p.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                   help=f"Transit speed in m/s (default {DEFAULT_SPEED})")
    p.add_argument("--alt", type=float, default=TAKEOFF_ALT,
                   help=f"Takeoff/cruise altitude in meters (default {TAKEOFF_ALT})")
    p.add_argument("--max-alt", type=float, default=10.0,
                   help="Maximum search altitude (default 10 — lower than mother since time is short)")
    p.add_argument("--recon-alt", type=float, default=RECON_ALT_DEFAULT,
                   help=f"Altitude to hover and observe (default {RECON_ALT_DEFAULT})")
    p.add_argument("--timer", type=int, default=MISSION_TIME_LIMIT_DEFAULT,
                   help=f"Mission time limit in seconds — RTL when expired (default {MISSION_TIME_LIMIT_DEFAULT})")

    p.add_argument("--weights", default="gol.pt",
                   help="YOLO weights file")
    p.add_argument("--conf", type=float, default=0.50,
                   help="YOLO confidence threshold (lower than mother — we WANT to see if it detects)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="YOLO input size")

    p.add_argument("--dry-run", action="store_true",
                   help="Camera only, no flight")
    p.add_argument("--sitl", action="store_true",
                   help="SITL mode")
    p.add_argument("--feed-port", type=int, default=5000,
                   help="Port for live browser feed (default 5000)")

    main(p.parse_args())
