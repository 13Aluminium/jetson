#!/usr/bin/env python3
"""
X Landing — Phase 2 Bench Test with Fake GPS
==============================================
Injects fake GPS into Pixhawk so it thinks it's flying at altitude.
Then uses GUIDED mode + velocity commands to test centering logic.
ArduPilot handles all motor mixing — we just say "go left/right/fwd/back".

PREREQUISITE — ONE-TIME PIXHAWK SETUP (via Mission Planner):
  1. GPS_TYPE   = 14   (MAVLink GPS)
  2. GPS_TYPE2  = 0    (disable second GPS if present)
  3. ARMING_CHECK = 0  (disable pre-arm checks for bench testing)
  4. Reboot Pixhawk after changing these params

  *** REMEMBER TO SET THESE BACK BEFORE REAL FLIGHT! ***
  For flight: GPS_TYPE=1, ARMING_CHECK=1

How this works:
  1. Connect to Pixhawk via USB-C
  2. Start sending fake GPS at 5Hz (3D fix, 10m altitude, stationary)
  3. Wait for EKF to accept the fake GPS (~15-30 seconds)
  4. Switch to GUIDED mode
  5. Arm
  6. Camera detects X → compute offset → send velocity commands
  7. ArduPilot figures out which motors to spin

Setup:
  - Mount camera pointing down
  - Connect Pixhawk via USB-C (/dev/ttyACM0)
  - REMOVE ALL PROPS!!!
  - Set GPS_TYPE=14, ARMING_CHECK=0 in Mission Planner first
  - Hold hardboard X target, move it around

Usage:
    python3 x_bench_fakegps.py --dry-run           # vision only, no Pixhawk
    python3 x_bench_fakegps.py                     # full test with fake GPS
    python3 x_bench_fakegps.py --log               # + save CSV
    python3 x_bench_fakegps.py --max-speed 0.3     # limit velocity (m/s)
    python3 x_bench_fakegps.py --headless           # SSH, no display
    python3 x_bench_fakegps.py --set-params         # auto-set GPS_TYPE=14 etc.

REMOVE PROPS FOR TESTING!

Transfer:
    scp x_bench_fakegps.py best_22.pt jetson@<ip>:~/
"""

import argparse
import time
import csv
import cv2
import os
import threading
import numpy as np
from datetime import datetime


# ---------------------------------------------------------------------------
# GStreamer pipeline (same as your working scripts)
# ---------------------------------------------------------------------------
def build_gstreamer_pipeline(
    sensor_mode=0,
    capture_width=3840,
    capture_height=2160,
    framerate=30,
    display_width=1920,
    display_height=1080,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-mode={sensor_mode} ! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"framerate=(fraction){framerate}/1, format=(string)NV12 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, "
        f"format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink max-buffers=2 drop=true"
    )


MODES = {
    "4k": {
        "sensor_mode": 0,
        "capture_width": 3840,
        "capture_height": 2160,
        "framerate": 30,
        "display_width": 1920,
        "display_height": 1080,
    },
    "1080p": {
        "sensor_mode": 1,
        "capture_width": 1920,
        "capture_height": 1080,
        "framerate": 60,
        "display_width": 1920,
        "display_height": 1080,
    },
}


# ---------------------------------------------------------------------------
# Load YOLO model
# ---------------------------------------------------------------------------
def load_model(weights="best_22.pt"):
    from ultralytics import YOLO

    if not os.path.exists(weights):
        print(f"[!] Model not found: {weights}")
        print(f"    scp best_22.pt jetson@<ip>:~/")
        exit(1)

    model = YOLO(weights)
    print(f"[+] Loaded: {weights}")
    print(f"    Classes: {model.names}")
    return model


# ---------------------------------------------------------------------------
# Find X center (from x_detect_guide.py)
# ---------------------------------------------------------------------------
def find_x_center_bbox(box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    return (x1 + x2) // 2, (y1 + y2) // 2


def find_x_center_refined(frame, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    h, w = frame.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)

    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return find_x_center_bbox(box)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    coords = np.column_stack(np.where(mask > 0))
    if len(coords) < 50:
        return find_x_center_bbox(box)

    return x1c + int(np.mean(coords[:, 1])), y1c + int(np.mean(coords[:, 0]))


# ---------------------------------------------------------------------------
# Compute offset
# ---------------------------------------------------------------------------
def compute_offset(cx, cy, frame_w, frame_h, deadzone=50):
    mid_x = frame_w // 2
    mid_y = frame_h // 2
    dx = cx - mid_x  # positive = X is RIGHT
    dy = cy - mid_y  # positive = X is BELOW

    if abs(dx) <= deadzone and abs(dy) <= deadzone:
        return dx, dy, True, "CENTERED"

    parts = []
    if dx < -deadzone:
        parts.append(f"LEFT ({abs(dx)}px)")
    elif dx > deadzone:
        parts.append(f"RIGHT ({abs(dx)}px)")

    if dy < -deadzone:
        parts.append(f"FWD ({abs(dy)}px)")
    elif dy > deadzone:
        parts.append(f"BACK ({abs(dy)}px)")

    return dx, dy, False, " + ".join(parts)


# ---------------------------------------------------------------------------
# Convert pixel offset to NED velocity
# ---------------------------------------------------------------------------
def pixel_offset_to_velocity(dx, dy, frame_w, frame_h, max_speed=0.5, deadzone=50):
    """
    Convert pixel offset to NED velocity (m/s).

    Camera pointing straight down:
      Camera +X (right in image) → drone should move right → +vy (East)
      Camera +Y (down in image)  → drone should move back  → -vx (South)

    *** FLIP SIGNS HERE IF BENCH TEST SHOWS WRONG DIRECTION ***
    """
    norm_x = dx / (frame_w / 2)
    norm_y = dy / (frame_h / 2)

    dz_x = deadzone / (frame_w / 2)
    dz_y = deadzone / (frame_h / 2)
    if abs(norm_x) < dz_x:
        norm_x = 0
    if abs(norm_y) < dz_y:
        norm_y = 0

    norm_x = max(-1.0, min(1.0, norm_x))
    norm_y = max(-1.0, min(1.0, norm_y))

    # Camera-to-NED mapping (edit if wrong direction!)
    vx = -norm_y * max_speed   # camera Y down = drone go back = -north
    vy = norm_x * max_speed    # camera X right = drone go right = +east
    vz = 0.0                   # no vertical for bench test

    return vx, vy, vz


# ===========================================================================
# PIXHAWK + FAKE GPS
# ===========================================================================

def connect_pixhawk(device, baud):
    from pymavlink import mavutil

    print(f"[MAV] Connecting to {device} at {baud}...")
    master = mavutil.mavlink_connection(device, baud=baud)
    print("[MAV] Waiting for heartbeat...")
    master.wait_heartbeat(timeout=30)
    master.target_component = 1
    print(f"[MAV] Connected! system={master.target_system}")
    return master


def wait_cmd_ack(master, command_id, timeout=5):
    start = time.time()
    while time.time() - start < timeout:
        try:
            msg = master.recv_match(type="COMMAND_ACK", blocking=True, timeout=1)
            if msg and msg.command == command_id:
                return msg
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Set parameters for fake GPS (can be done from script)
# ---------------------------------------------------------------------------
def set_param(master, name, value):
    from pymavlink import mavutil

    print(f"[MAV] Setting {name} = {value}...")
    master.mav.param_set_send(
        master.target_system,
        master.target_component,
        name.encode('utf-8'),
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )
    # Wait for param ack
    time.sleep(0.5)
    msg = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=3)
    if msg and msg.param_id.replace('\x00', '') == name:
        print(f"    {name} confirmed = {msg.param_value}")
        return True
    else:
        print(f"    {name} — no confirmation (may need reboot)")
        return False


def setup_fakegps_params(master):
    """Set required params for fake GPS. Requires Pixhawk reboot after!"""
    print("\n[MAV] Setting fake GPS parameters...")
    print("      *** Pixhawk must be REBOOTED after this! ***\n")

    set_param(master, 'GPS_TYPE', 14)      # MAVLink GPS
    set_param(master, 'GPS_TYPE2', 0)      # Disable GPS2
    set_param(master, 'ARMING_CHECK', 0)   # Disable pre-arm checks

    # Also helpful for bench:
    set_param(master, 'EK3_SRC1_POSXY', 1)  # GPS for position
    set_param(master, 'EK3_SRC1_VELXY', 1)  # GPS for velocity
    set_param(master, 'EK3_SRC1_POSZ', 1)   # GPS for altitude

    print("\n[MAV] Parameters set. Now REBOOT Pixhawk!")
    print("      In Mission Planner: Actions → Reboot")
    print("      Or power-cycle the Pixhawk\n")
    print("      Then run this script again WITHOUT --set-params\n")


# ---------------------------------------------------------------------------
# Fake GPS sender (runs in background thread)
# ---------------------------------------------------------------------------
class FakeGPS:
    """
    Sends GPS_INPUT messages at 5Hz to make Pixhawk think
    it has a 3D GPS fix at a fixed location and altitude.
    """

    def __init__(self, master, lat=33.77, lon=-118.19, alt=10.0):
        """
        lat/lon: any valid coordinate (Long Beach, CA by default)
        alt: simulated altitude in meters
        """
        self.master = master
        self.lat = int(lat * 1e7)     # degrees * 1E7
        self.lon = int(lon * 1e7)     # degrees * 1E7
        self.alt = alt                # meters
        self.running = False
        self.thread = None
        self.gps_ok = False           # set True once EKF converges
        self._lock = threading.Lock()

        # GPS time calculation
        from datetime import datetime
        self.GPS_EPOCH = datetime(1980, 1, 6)

    def _get_gps_time(self):
        now = datetime.utcnow()
        delta = now - self.GPS_EPOCH
        gps_week = delta.days // 7
        gps_week_ms = ((delta.days % 7) * 86400 + delta.seconds) * 1000
        return gps_week, gps_week_ms

    def _send_loop(self):
        while self.running:
            try:
                gps_week, gps_week_ms = self._get_gps_time()

                self.master.mav.gps_input_send(
                    int(time.time() * 1e6),   # time_usec
                    0,                         # gps_id
                    # ignore_flags bitmask (raw values — works on all pymavlink versions)
                    # bit 3 = speed_accuracy (8)
                    # bit 4 = horiz_accuracy (16)
                    # bit 5 = vert_accuracy  (32)
                    8 | 16 | 32,              # = 56
                    gps_week_ms,              # time_week_ms
                    gps_week,                 # time_week
                    3,                         # fix_type: 3D fix
                    self.lat,                  # lat (deg * 1E7)
                    self.lon,                  # lon (deg * 1E7)
                    self.alt,                  # alt (m AMSL)
                    0.8,                       # hdop
                    0.8,                       # vdop
                    0.0,                       # vn (m/s)
                    0.0,                       # ve (m/s)
                    0.0,                       # vd (m/s)
                    0.0,                       # speed_accuracy
                    0.5,                       # horiz_accuracy (m)
                    1.0,                       # vert_accuracy (m)
                    12,                        # satellites_visible
                )
            except Exception as e:
                print(f"[GPS] Send error: {e}")

            time.sleep(0.2)  # 5 Hz

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._send_loop, daemon=True)
        self.thread.start()
        print(f"[GPS] Fake GPS started: lat={self.lat/1e7:.4f}, "
              f"lon={self.lon/1e7:.4f}, alt={self.alt}m")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("[GPS] Fake GPS stopped")

    def wait_for_ekf(self, timeout=60):
        """Wait for EKF to converge and accept our fake GPS."""
        from pymavlink import mavutil

        print(f"[GPS] Waiting for EKF to accept fake GPS (up to {timeout}s)...")
        print(f"      Keep Pixhawk still on the bench!")
        start = time.time()

        while time.time() - start < timeout:
            # Check for GPS_RAW_INT to see if Pixhawk accepts our GPS
            msg = self.master.recv_match(
                type=['GPS_RAW_INT', 'EKF_STATUS_REPORT'],
                blocking=True, timeout=2
            )

            if msg:
                if msg.get_type() == 'GPS_RAW_INT':
                    fix = msg.fix_type
                    sats = msg.satellites_visible
                    elapsed = time.time() - start
                    print(f"\r[GPS] {elapsed:.0f}s — fix_type={fix}, "
                          f"sats={sats}     ", end="", flush=True)

                    if fix >= 3:
                        print(f"\n[GPS] 3D fix obtained!")
                        # Give EKF a few more seconds to settle
                        print("[GPS] Waiting 10s for EKF to settle...")
                        time.sleep(10)
                        self.gps_ok = True
                        return True

                elif msg.get_type() == 'EKF_STATUS_REPORT':
                    flags = msg.flags
                    # Check if position is good (bit 0 and 2)
                    pos_ok = bool(flags & 0x01)
                    vel_ok = bool(flags & 0x02)
                    if pos_ok and vel_ok:
                        pass  # Good, keep waiting for GPS fix

        print(f"\n[GPS] Timeout waiting for EKF ({timeout}s)")
        print("      Check: GPS_TYPE=14? Rebooted after setting?")
        return False


# ---------------------------------------------------------------------------
# GUIDED mode + arm + velocity commands
# ---------------------------------------------------------------------------
def set_guided_mode(master):
    from pymavlink import mavutil

    print("[MAV] Setting GUIDED mode...")
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        4  # GUIDED for ArduCopter
    )
    time.sleep(2)

    # Verify mode
    hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
    if hb and hb.custom_mode == 4:
        print("[MAV] GUIDED mode confirmed!")
        return True
    else:
        mode = hb.custom_mode if hb else "?"
        print(f"[MAV] Mode is {mode}, expected 4 (GUIDED)")
        return False


def arm_vehicle(master):
    from pymavlink import mavutil

    print("[MAV] Arming (normal arm, not force)...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0)  # normal arm, no force

    ack = wait_cmd_ack(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
    if ack and ack.result == 0:
        print("[MAV] Armed!")
        return True

    # If normal arm fails, try force arm
    print("[MAV] Normal arm failed, trying force arm (21196)...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 21196, 0, 0, 0, 0, 0)

    ack = wait_cmd_ack(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
    if ack and ack.result == 0:
        print("[MAV] Force armed!")
        return True

    print(f"[MAV] Arm failed: {ack.result if ack else 'no response'}")
    return False


def disarm_vehicle(master):
    from pymavlink import mavutil

    print("[MAV] Disarming...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, 21196, 0, 0, 0, 0, 0)

    ack = wait_cmd_ack(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
    if ack and ack.result == 0:
        print("[MAV] Disarmed!")
    else:
        print(f"[MAV] Disarm: {ack.result if ack else 'no response'}")


def send_velocity_ned(master, vx, vy, vz):
    """Send velocity in NED frame. vx=north, vy=east, vz=down."""
    from pymavlink import mavutil

    type_mask = (
        (1 << 0) | (1 << 1) | (1 << 2) |   # ignore position
        # bits 3,4,5 = 0 → USE velocity
        (1 << 6) | (1 << 7) | (1 << 8) |    # ignore acceleration
        (1 << 9) | (1 << 10)                 # ignore yaw
    )

    master.mav.set_position_target_local_ned_send(
        0,                                     # time_boot_ms
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        0, 0, 0,           # position (ignored)
        vx, vy, vz,        # velocity m/s
        0, 0, 0,           # acceleration (ignored)
        0, 0               # yaw, yaw_rate (ignored)
    )


# ---------------------------------------------------------------------------
# CSV Logger
# ---------------------------------------------------------------------------
class BenchLogger:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.file = None
        self.writer = None
        if enabled:
            fname = f"bench_fakegps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.file = open(fname, 'w', newline='')
            self.writer = csv.writer(self.file)
            self.writer.writerow([
                'timestamp', 'frame_num',
                'x_detected', 'conf',
                'cx', 'cy', 'dx_px', 'dy_px', 'centered',
                'vx_ned', 'vy_ned', 'vz_ned',
                'direction', 'fps'
            ])
            print(f"[LOG] Logging to: {fname}")

    def log(self, frame_num, detected, conf, cx, cy,
            dx, dy, centered, vx, vy, vz, direction, fps):
        if not self.enabled:
            return
        self.writer.writerow([
            datetime.now().isoformat(), frame_num,
            detected, f"{conf:.3f}",
            cx, cy, dx, dy, centered,
            f"{vx:.4f}", f"{vy:.4f}", f"{vz:.4f}",
            direction, f"{fps:.1f}"
        ])
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


# ---------------------------------------------------------------------------
# Main bench test loop
# ---------------------------------------------------------------------------
def run_bench(args):
    print("=" * 65)
    print("  X LANDING — PHASE 2 BENCH TEST (FAKE GPS + GUIDED)")
    print("=" * 65)
    if args.dry_run:
        print("  MODE: DRY RUN (vision only)")
    else:
        print(f"  MODE: LIVE (Pixhawk @ {args.device})")
        print(f"  FAKE GPS: lat={args.lat}, lon={args.lon}, alt={args.alt}m")
    print(f"  MAX SPEED: {args.max_speed} m/s")
    print(f"  DEADZONE: {args.deadzone}px")
    print(f"  CENTER: {'refined' if args.refine else 'bbox'}")
    print("=" * 65)

    # Load model
    print("\n[*] Loading YOLO model...")
    model = load_model(args.weights)

    # Connect Pixhawk + start fake GPS
    master = None
    fake_gps = None
    armed = False

    if not args.dry_run:
        master = connect_pixhawk(args.device, args.baud)

        # Start fake GPS
        fake_gps = FakeGPS(master, lat=args.lat, lon=args.lon, alt=args.alt)
        fake_gps.start()

        # Wait for EKF
        ekf_ok = fake_gps.wait_for_ekf(timeout=args.ekf_timeout)
        if not ekf_ok:
            print("[!] EKF did not converge. Check GPS_TYPE=14 and reboot.")
            print("    Run with --set-params first if you haven't.")
            fake_gps.stop()
            return

        # Set GUIDED mode
        guided_ok = set_guided_mode(master)
        if not guided_ok:
            print("[!] Could not set GUIDED mode")

        # Arm
        armed = arm_vehicle(master)
        if not armed:
            print("[!] Could not arm. Motors won't respond to velocity.")
            print("    Check ARMING_CHECK=0 is set and Pixhawk rebooted.")

    # Open camera
    mode_cfg = MODES[args.mode]
    pipeline = build_gstreamer_pipeline(**mode_cfg)
    infer_w = mode_cfg["display_width"]
    infer_h = mode_cfg["display_height"]
    print(f"[*] Opening camera: {args.mode} → {infer_w}x{infer_h}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[!] Failed to open camera.")
        if fake_gps:
            fake_gps.stop()
        if armed:
            disarm_vehicle(master)
        return

    # Logger
    logger = BenchLogger(args.log)

    # Display
    view_w, view_h = 960, 540
    show_display = not args.headless
    if show_display:
        window_name = "X Landing — Fake GPS Bench Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, view_w, view_h)

    # Loop state
    fps_count = 0
    fps_start = time.time()
    display_fps = 0.0
    frame_num = 0
    last_cmd_time = 0
    cmd_interval = 0.1  # 10 Hz velocity commands

    last_detection_time = time.time()
    no_detect_timeout = 2.0

    method_tag = "REFINED" if args.refine else "BBOX"
    print(f"\n[*] Running! Move your X target around.")
    print(f"    ArduPilot will handle motor mixing via GUIDED velocity.")
    print(f"    Press 'q' to quit, 's' for snapshot")
    print("-" * 65)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame_num += 1
            frame_h, frame_w = frame.shape[:2]
            now = time.time()

            # --- YOLO inference ---
            results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

            # --- Find best detection ---
            best_box = None
            best_conf = 0.0
            best_result = None
            det_count = 0

            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < args.conf:
                        continue
                    det_count += 1
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box
                        best_result = result

            # --- Compute offset + velocity ---
            cx, cy = 0, 0
            dx, dy = 0, 0
            centered = False
            direction_str = "NO TARGET"
            vx, vy, vz = 0.0, 0.0, 0.0

            if best_box is not None:
                last_detection_time = now

                if args.refine:
                    cx, cy = find_x_center_refined(frame, best_box)
                else:
                    cx, cy = find_x_center_bbox(best_box)

                dx, dy, centered, direction_str = compute_offset(
                    cx, cy, frame_w, frame_h, args.deadzone
                )

                if not centered:
                    vx, vy, vz = pixel_offset_to_velocity(
                        dx, dy, frame_w, frame_h,
                        max_speed=args.max_speed,
                        deadzone=args.deadzone
                    )

            else:
                # No detection — zero velocity
                if now - last_detection_time > no_detect_timeout:
                    direction_str = "NO TARGET (zeroed)"

            # --- Send velocity to Pixhawk ---
            if not args.dry_run and master and armed:
                if now - last_cmd_time >= cmd_interval:
                    send_velocity_ned(master, vx, vy, vz)
                    last_cmd_time = now

            # --- Log ---
            logger.log(
                frame_num, det_count > 0, best_conf,
                cx, cy, dx, dy, centered,
                vx, vy, vz, direction_str, display_fps
            )

            # --- Terminal output ---
            if det_count > 0:
                vel_str = f"vel=({vx:+.2f},{vy:+.2f},{vz:+.2f})"
                if centered:
                    print(f"\r[CENTERED] conf={best_conf:.2f} "
                          f"X@({cx},{cy}) HOLD "
                          f"FPS={display_fps:.1f}           ", end="", flush=True)
                else:
                    print(f"\r[{direction_str}] {vel_str} "
                          f"FPS={display_fps:.1f}    ", end="", flush=True)
            else:
                print(f"\rSearching... FPS={display_fps:.1f}    ",
                      end="", flush=True)

            # --- FPS ---
            fps_count += 1
            elapsed = now - fps_start
            if elapsed >= 1.0:
                display_fps = fps_count / elapsed
                fps_count = 0
                fps_start = now

            # --- Display ---
            if show_display:
                display_frame = cv2.resize(frame, (view_w, view_h))
                scale_x = view_w / frame_w
                scale_y = view_h / frame_h

                # Frame center crosshair
                smid_x, smid_y = view_w // 2, view_h // 2
                cv2.line(display_frame, (smid_x - 30, smid_y),
                         (smid_x + 30, smid_y), (200, 200, 200), 1)
                cv2.line(display_frame, (smid_x, smid_y - 30),
                         (smid_x, smid_y + 30), (200, 200, 200), 1)
                cv2.circle(display_frame, (smid_x, smid_y), 6, (200, 200, 200), 1)

                # Deadzone rectangle
                dz_x = int(args.deadzone * scale_x)
                dz_y = int(args.deadzone * scale_y)
                cv2.rectangle(display_frame,
                              (smid_x - dz_x, smid_y - dz_y),
                              (smid_x + dz_x, smid_y + dz_y),
                              (100, 100, 100), 1)

                # Draw detections
                for result in results:
                    for box in result.boxes:
                        conf_val = float(box.conf[0])
                        if conf_val < args.conf:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_name = result.names[int(box.cls[0])]
                        dx1, dy1 = int(x1 * scale_x), int(y1 * scale_y)
                        dx2, dy2 = int(x2 * scale_x), int(y2 * scale_y)
                        cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2),
                                      (0, 255, 0), 2)
                        label = f"{cls_name} {conf_val:.2f}"
                        lsz, _ = cv2.getTextSize(label,
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(display_frame, (dx1, dy1 - lsz[1] - 10),
                                      (dx1 + lsz[0], dy1), (0, 255, 0), -1)
                        cv2.putText(display_frame, label, (dx1, dy1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # X center + guidance line
                if best_box is not None:
                    dcx = int(cx * scale_x)
                    dcy = int(cy * scale_y)
                    dot_color = (0, 255, 0) if centered else (0, 0, 255)
                    cv2.circle(display_frame, (dcx, dcy), 8, dot_color, -1)
                    cv2.circle(display_frame, (dcx, dcy), 10, (255, 255, 255), 2)
                    cv2.line(display_frame, (dcx, dcy),
                             (smid_x, smid_y), dot_color, 2)

                    # Velocity arrow
                    arrow_scale = 200
                    ax = smid_x + int(vy * arrow_scale)
                    ay = smid_y - int(vx * arrow_scale)  # NED north = up on screen
                    cv2.arrowedLine(display_frame, (smid_x, smid_y),
                                    (ax, ay), (0, 255, 255), 3, tipLength=0.3)

                # Guidance text
                if centered:
                    guide_text = "*** CENTERED — HOLDING ***"
                    guide_color = (0, 255, 0)
                elif det_count > 0:
                    guide_text = direction_str
                    guide_color = (0, 165, 255)
                else:
                    guide_text = "Searching..."
                    guide_color = (150, 150, 150)

                cv2.putText(display_frame, guide_text, (10, view_h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, guide_color, 2)

                # Velocity readout
                vel_text = f"NED vel: ({vx:+.2f}, {vy:+.2f}, {vz:+.2f}) m/s"
                cv2.putText(display_frame, vel_text, (10, view_h - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if det_count > 0:
                    off_text = f"offset: ({dx:+d}, {dy:+d})px"
                    cv2.putText(display_frame, off_text, (10, view_h - 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Status bar
                mode_str = "DRY RUN" if args.dry_run else "GUIDED+GPS"
                armed_str = "ARMED" if armed else "DISARMED"
                info = (f"FPS:{display_fps:.1f} | X:{det_count} | "
                        f"{method_tag} | {mode_str} | {armed_str}")
                bar_color = (0, 255, 0) if armed else (200, 200, 200)
                cv2.putText(display_frame, info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)

                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord('s'):
                    fname = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    print(f"\n[+] Saved: {fname}")

    except KeyboardInterrupt:
        print("\n[*] Stopped by user.")

    finally:
        # Cleanup: zero velocity, disarm, stop fake GPS
        if not args.dry_run and master:
            print("\n[*] Cleaning up...")
            send_velocity_ned(master, 0, 0, 0)
            time.sleep(0.5)
            disarm_vehicle(master)

        if fake_gps:
            fake_gps.stop()

        logger.close()
        cap.release()
        if show_display:
            cv2.destroyAllWindows()
        print("[*] Bench test complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="X Landing — Phase 2 Bench Test with Fake GPS (PROPS OFF!)")

    # Detection
    parser.add_argument("--weights", default="best_22.pt")
    parser.add_argument("--mode", choices=["4k", "1080p"], default="1080p")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--refine", action="store_true",
                        help="Green segmentation for precise X center")
    parser.add_argument("--deadzone", type=int, default=50)

    # Pixhawk
    parser.add_argument("--device", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--max-speed", type=float, default=0.3,
                        help="Max velocity m/s (default: 0.3)")

    # Fake GPS
    parser.add_argument("--lat", type=float, default=33.77,
                        help="Fake GPS latitude (default: 33.77 Long Beach)")
    parser.add_argument("--lon", type=float, default=-118.19,
                        help="Fake GPS longitude (default: -118.19)")
    parser.add_argument("--alt", type=float, default=10.0,
                        help="Fake GPS altitude meters (default: 10)")
    parser.add_argument("--ekf-timeout", type=int, default=60,
                        help="Max seconds to wait for EKF (default: 60)")

    # Modes
    parser.add_argument("--dry-run", action="store_true",
                        help="Vision only, no Pixhawk")
    parser.add_argument("--headless", action="store_true",
                        help="No display (SSH)")
    parser.add_argument("--log", action="store_true",
                        help="Save CSV log")
    parser.add_argument("--set-params", action="store_true",
                        help="Set GPS_TYPE=14 etc. on Pixhawk (then reboot)")

    args = parser.parse_args()

    # --- Set params mode ---
    if args.set_params:
        master = connect_pixhawk(args.device, args.baud)
        setup_fakegps_params(master)
        exit(0)

    # --- Normal run ---
    if not args.dry_run:
        print("\n" + "!" * 65)
        print("  PREREQUISITES:")
        print("    1. GPS_TYPE = 14 (set via --set-params or Mission Planner)")
        print("    2. ARMING_CHECK = 0")
        print("    3. Pixhawk REBOOTED after setting params")
        print("    4. PROPS REMOVED!!!")
        print("!" * 65)
        resp = input("\n  All done? Props removed? (y/n): ").strip().lower()
        if resp != 'y':
            print("[*] Aborted.")
            exit(0)

    run_bench(args)
