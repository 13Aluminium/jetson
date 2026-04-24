#!/usr/bin/env python3
"""
flight_utils.py — Shared utilities for all X-Landing flight scripts
=====================================================================
Architecture:
    MAVProxy runs separately, connected to the Pixhawk (or SITL).
    Our scripts connect to MAVProxy via UDP on port 14551.
    
    Real flight (2 SSH terminals to Jetson):
      Terminal 1: mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 --out=udp:127.0.0.1:14551
      Terminal 2: python3 <script>.py
    
    SITL testing (on Mac/Linux with ArduPilot installed):
      Terminal 1: sim_vehicle.py -v ArduCopter --console --map --out=udp:127.0.0.1:14551
      Terminal 2: python3 <script>.py --sitl

Why MAVProxy?
    - Battle-tested, handles all MAVLink complexity
    - You can type emergency commands (mode RTL) in MAVProxy terminal
    - Same script works for SITL and real flight
    - MAVProxy handles heartbeats, parameter management, etc.

Hardware: Jetson Orin Nano + IMX477 Camera + Pixhawk 6C
"""

import os
import sys
import time
import signal
import math
import subprocess
import cv2
from datetime import datetime
from pymavlink import mavutil


# ===========================================================================
# CONSTANTS
# ===========================================================================
MAVPROXY_UDP = "udp:127.0.0.1:14551"

# ArduCopter flight modes
MODE_STABILIZE = 0
MODE_GUIDED = 4
MODE_LOITER = 5
MODE_RTL = 6
MODE_LAND = 9

TAKEOFF_ALT = 5.0  # meters

# Camera
FRAME_W = 1920
FRAME_H = 1080
HFOV_DEG = 73.0

# YOLO
DEFAULT_WEIGHTS = "best_22.pt"
DEFAULT_CONF = 0.50
DEFAULT_IMGSZ = 640

# Timeouts
HEARTBEAT_TIMEOUT = 10.0
ARM_TIMEOUT = 15.0
TAKEOFF_TIMEOUT = 30.0
CMD_ACK_TIMEOUT = 10.0


# ===========================================================================
# FLIGHT CONTROLLER (connects to MAVProxy via UDP)
# ===========================================================================
class FlightController:
    def __init__(self, connection_string=MAVPROXY_UDP):
        self.conn_str = connection_string
        self.master = None
        self.armed = False
        self.mode = 0
        self.mode_name = "UNKNOWN"
        self.alt = 0.0
        self.lat = 0.0
        self.lon = 0.0
        self.gps_fix = 0
        self.satellites = 0
        self.battery_pct = -1
        self.heading = 0
        self.last_hb_recv = 0

    def connect(self):
        print(f"[MAV] Connecting to MAVProxy at {self.conn_str}...")
        print(f"      (MAVProxy must be running in another terminal!)")
        self.master = mavutil.mavlink_connection(self.conn_str)
        print("[MAV] Waiting for heartbeat...")
        hb = self.master.wait_heartbeat(timeout=30)
        if hb is None:
            print("[!] No heartbeat! Is MAVProxy running?")
            print()
            print("  Real flight:  mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 --out=udp:127.0.0.1:14551")
            print("  SITL:         sim_vehicle.py -v ArduCopter --console --map --out=udp:127.0.0.1:14551")
            sys.exit(1)
        self.last_hb_recv = time.time()
        self.master.target_component = 1
        print(f"[MAV] Connected! sysid={self.master.target_system}")
        self._request_streams()
        time.sleep(2)
        self.poll()
        return self.master

    def _request_streams(self):
        for stream, rate in [
            (mavutil.mavlink.MAV_DATA_STREAM_ALL, 4),
            (mavutil.mavlink.MAV_DATA_STREAM_POSITION, 10),
        ]:
            self.master.mav.request_data_stream_send(
                self.master.target_system, self.master.target_component,
                stream, rate, 1)

    def poll(self):
        status_texts = []
        while True:
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                break
            mtype = msg.get_type()
            if mtype == 'HEARTBEAT' and msg.get_srcSystem() == self.master.target_system:
                self.last_hb_recv = time.time()
                self.armed = bool(msg.base_mode & 128)
                self.mode = msg.custom_mode
                mode_map = {0:"STABILIZE", 2:"ALT_HOLD", 3:"AUTO", 4:"GUIDED",
                            5:"LOITER", 6:"RTL", 7:"CIRCLE", 9:"LAND", 16:"POSHOLD"}
                self.mode_name = mode_map.get(self.mode, f"MODE_{self.mode}")
            elif mtype == 'GLOBAL_POSITION_INT':
                self.alt = msg.relative_alt / 1000.0
                self.lat = msg.lat / 1e7
                self.lon = msg.lon / 1e7
                self.heading = msg.hdg / 100.0 if msg.hdg != 65535 else 0
            elif mtype == 'GPS_RAW_INT':
                self.gps_fix = msg.fix_type
                self.satellites = msg.satellites_visible
            elif mtype == 'SYS_STATUS':
                if msg.battery_remaining >= 0:
                    self.battery_pct = msg.battery_remaining
            elif mtype == 'STATUSTEXT':
                text = msg.text.strip('\x00').strip()
                if text:
                    status_texts.append(text)
                    print(f"  [PIXHAWK] {text}")
        return status_texts

    def check_alive(self):
        return (time.time() - self.last_hb_recv) < HEARTBEAT_TIMEOUT if self.last_hb_recv else False

    def wait_ack(self, cmd_id, timeout=CMD_ACK_TIMEOUT):
        start = time.time()
        while time.time() - start < timeout:
            msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=1)
            if msg and msg.command == cmd_id:
                return msg
            self.poll()
        return None

    # ── Mode ──
    def set_mode(self, mode_id):
        mode_map = {MODE_GUIDED:"GUIDED", MODE_LAND:"LAND", MODE_RTL:"RTL", MODE_LOITER:"LOITER"}
        name = mode_map.get(mode_id, str(mode_id))
        print(f"[CMD] mode {name}")
        self.master.set_mode(mode_id)
        start = time.time()
        while time.time() - start < 5.0:
            self.poll()
            if self.mode == mode_id:
                print(f"[OK]  Mode → {name}")
                return True
            time.sleep(0.2)
        print(f"[!]   Mode change not confirmed (current: {self.mode_name})")
        return False

    def set_guided(self): return self.set_mode(MODE_GUIDED)
    def set_rtl(self):
        print("[CMD] >>> RTL <<<")
        return self.set_mode(MODE_RTL)
    def set_land(self):
        print("[CMD] >>> LAND <<<")
        return self.set_mode(MODE_LAND)

    # ── Arm / Disarm ──
    def arm(self):
        print("[CMD] arm throttle")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
        ack = self.wait_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
        if ack and ack.result == 0:
            self.armed = True
            print("[OK]  Armed!")
            return True
        print(f"[!]   Arm FAILED: {ack.result if ack else 'no response'}")
        return False

    def disarm(self, force=False):
        p2 = 21196 if force else 0
        print(f"[CMD] disarm {'(forced)' if force else ''}")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, p2, 0, 0, 0, 0, 0)
        ack = self.wait_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
        if ack and ack.result == 0:
            self.armed = False
            print("[OK]  Disarmed!")
            return True
        return False

    # ── Takeoff ──
    def takeoff(self, alt_m):
        print(f"[CMD] takeoff {alt_m}m")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, alt_m)
        ack = self.wait_ack(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF)
        if ack and ack.result == 0:
            print(f"[OK]  Takeoff → {alt_m}m")
            return True
        print(f"[!]   Takeoff FAILED: {ack.result if ack else 'no response'}")
        return False

    def wait_alt(self, target, tol=1.0, timeout=TAKEOFF_TIMEOUT):
        print(f"[...] Climbing to {target}m...")
        start = time.time()
        while time.time() - start < timeout:
            self.poll()
            if self.alt >= target - tol:
                print(f"\n[OK]  Reached {self.alt:.1f}m")
                return True
            print(f"\r      {self.alt:.1f}m / {target}m   ", end="", flush=True)
            time.sleep(0.3)
        print(f"\n[!]   Timeout at {self.alt:.1f}m")
        return False

    # ── Movement ──
    def velocity_ned(self, vn, ve, vd):
        """NED velocity. Re-send every ~1s. Vehicle stops after 3s."""
        self.master.mav.set_position_target_local_ned_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111, 0,0,0, vn,ve,vd, 0,0,0, 0,0)

    def velocity_body(self, vx_fwd, vy_right, vz_down):
        """Body-frame velocity. +x=forward, +y=right, +z=down."""
        self.master.mav.set_position_target_local_ned_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000111, 0,0,0, vx_fwd,vy_right,vz_down, 0,0,0, 0,0)

    def move_body(self, fwd_m, right_m, down_m):
        """Position offset relative to current pos + heading."""
        print(f"[CMD] move fwd={fwd_m:+.1f} right={right_m:+.1f} down={down_m:+.1f}")
        self.master.mav.set_position_target_local_ned_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000111111111000, fwd_m,right_m,down_m, 0,0,0, 0,0,0, 0,0)

    def stop(self):
        self.velocity_ned(0, 0, 0)

    # ── Wait ──
    def wait_disarmed(self, timeout=60):
        print("[...] Waiting for landing...")
        start = time.time()
        while time.time() - start < timeout:
            self.poll()
            if not self.armed:
                print("\n[OK]  Landed (disarmed).")
                return True
            print(f"\r      Alt: {self.alt:.1f}m   ", end="", flush=True)
            time.sleep(0.5)
        print(f"\n[!]   Still armed after {timeout}s")
        return False

    # ── Pre-flight ──
    def preflight(self):
        print("\n" + "=" * 55)
        print("  PRE-FLIGHT CHECKS")
        print("=" * 55)
        for _ in range(10):
            self.poll()
            time.sleep(0.2)
        ok = True
        if self.gps_fix < 3:
            print(f"  [FAIL] GPS fix: {self.gps_fix} (need ≥3)"); ok = False
        else:
            print(f"  [OK]   GPS: fix={self.gps_fix} sats={self.satellites}")
        if 0 <= self.battery_pct < 25:
            print(f"  [FAIL] Battery: {self.battery_pct}%"); ok = False
        elif self.battery_pct >= 25:
            print(f"  [OK]   Battery: {self.battery_pct}%")
        else:
            print(f"  [WARN] Battery: unknown")
        if abs(self.lat) < 0.001 and abs(self.lon) < 0.001:
            print(f"  [FAIL] No GPS position"); ok = False
        else:
            print(f"  [OK]   Pos: {self.lat:.6f}, {self.lon:.6f}")
        print(f"  [INFO] Mode: {self.mode_name} | Armed: {self.armed}")
        print("=" * 55)
        print(f"  {'PASSED ✓' if ok else 'FAILED ✗ — DO NOT FLY'}")
        print("=" * 55 + "\n")
        return ok

    def close(self):
        if self.master:
            self.master.close()


# ===========================================================================
# CAMERA
# ===========================================================================
def _run(cmd, check=False):
    """Run a shell command, return (returncode, stdout). Swallow errors."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True,
                           text=True, timeout=10)
        return r.returncode, r.stdout.strip()
    except Exception:
        return -1, ""

def free_camera():
    """Kill anything holding /dev/video* and restart nvargus-daemon."""
    print("[PRE] Freeing camera resources...")

    # 1) Find and kill processes locking /dev/video*
    rc, out = _run("fuser /dev/video0 2>/dev/null")
    if rc == 0 and out:
        pids = out.split()
        my_pid = str(os.getpid())
        stale = [p for p in pids if p != my_pid]
        if stale:
            print(f"[PRE]   /dev/video0 locked by PIDs: {' '.join(stale)}")
            for pid in stale:
                _, pname = _run(f"ps -p {pid} -o comm= 2>/dev/null")
                print(f"[PRE]   Killing PID {pid} ({pname or 'unknown'})...")
                _run(f"kill -9 {pid}")
            time.sleep(1)
            print("[PRE]   Stale processes cleared.")
        else:
            print("[PRE]   /dev/video0 held by us only — OK.")
    else:
        print("[PRE]   /dev/video0 is free.")

    # 2) Kill any leftover nvarguscamerasrc / gst-launch processes
    for proc in ["nvarguscamerasrc", "gst-launch"]:
        _run(f"pkill -9 -f {proc} 2>/dev/null")


def _ensure_nvargus():
    """Make sure nvargus-daemon is running and its socket is ready."""

    # Check if already running and healthy
    rc, _ = _run("pgrep -x nvargus-daemon")
    if rc == 0:
        print("[PRE]   nvargus-daemon already running.")
        return True

    # Not running — start it
    print("[PRE]   Starting nvargus-daemon...")

    # Try systemctl first (preferred — manages it as a service)
    rc, _ = _run("sudo systemctl start nvargus-daemon 2>/dev/null")
    if rc == 0:
        # Wait for daemon to initialize its socket
        for i in range(8):
            time.sleep(1)
            rc2, _ = _run("pgrep -x nvargus-daemon")
            if rc2 == 0:
                print(f"[PRE]   nvargus-daemon started via systemctl ({i+1}s).")
                return True
        print("[PRE]   ⚠ systemctl started but daemon not responding")

    # Fallback: run directly (daemonizes itself)
    print("[PRE]   Trying direct launch...")
    _run("sudo nvargus-daemon &>/dev/null &", check=False)

    # Wait for it to come up
    for i in range(8):
        time.sleep(1)
        rc, _ = _run("pgrep -x nvargus-daemon")
        if rc == 0:
            print(f"[PRE]   nvargus-daemon started directly ({i+1}s).")
            return True

    print("[PRE]   ⚠ Could not start nvargus-daemon!")
    return False

def camera_precheck(sitl=False):
    """
    Run all camera prechecks. Returns True if camera should be available.
    Call this before open_camera() for better diagnostics.
    """
    if sitl:
        return True

    print("\n" + "-" * 55)
    print("  CAMERA PRE-CHECKS")
    print("-" * 55)
    ok = True

    # Check /dev/video0 exists
    if os.path.exists("/dev/video0"):
        print("  [OK]   /dev/video0 exists")
    else:
        print("  [FAIL] /dev/video0 not found!")
        print("         Check ribbon cable, run: sudo media-ctl -p")
        ok = False

    # Check GStreamer available
    rc, _ = _run("gst-inspect-1.0 nvarguscamerasrc 2>/dev/null")
    if rc == 0:
        print("  [OK]   GStreamer nvarguscamerasrc plugin found")
    else:
        print("  [WARN] nvarguscamerasrc plugin not found")
        ok = False

    # Free stale locks (does NOT touch nvargus-daemon)
    free_camera()

    # Ensure nvargus-daemon is running (starts only if needed)
    _ensure_nvargus()

    # Give daemon socket a moment to stabilize
    time.sleep(1)

    print("-" * 55)
    print(f"  {'CAMERA READY ✓' if ok else 'CAMERA ISSUES ✗ — will try anyway'}")
    print("-" * 55 + "\n")
    return ok


def open_camera(sitl=False):
    if sitl:
        print("[CAM] SITL mode — trying webcam...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("[CAM] Webcam ready.")
            return cap
        print("[CAM] No webcam — camera features disabled in SITL.")
        return None

    # Run prechecks (free stale locks, restart daemon, verify device)
    camera_precheck(sitl=False)

    # Pipeline notes:
    #   - sensor-mode=1 = 1920x1080 (native, no scaling overhead)
    #   - nvvidconv does NV12→BGRx in hardware (GPU/VIC)
    #   - videoconvert does BGRx→BGR in software (CPU) — this is the bottleneck
    #   - drop=true + max-buffers=1 ensures we always get the latest frame,
    #     never a stale queued one (critical for landing guidance)
    pipeline = (
        "nvarguscamerasrc sensor-mode=1 ! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, "
        "framerate=30/1, format=NV12 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=1920, height=1080, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink max-buffers=1 drop=true sync=false"
    )
    print(f"[CAM] Opening IMX477 1920x1080...")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[!] Camera failed on first attempt — restarting daemon and retrying...")
        _run("sudo systemctl stop nvargus-daemon 2>/dev/null")
        _run("sudo killall -9 nvargus-daemon 2>/dev/null")
        time.sleep(2)
        free_camera()
        _ensure_nvargus()
        time.sleep(3)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print("[!] Camera failed again! Check hardware:")
            print("    1. Ribbon cable seated properly?")
            print("    2. sudo media-ctl -p")
            print("    3. ls /dev/video*")
            print("    4. sudo systemctl restart nvargus-daemon && sleep 5")
            return None

    # Flush initial dark/garbage frames
    for _ in range(15): cap.read()

    # ── Measure ACTUAL FPS ──
    # CAP_PROP_FPS lies (returns pipeline requested rate, not real throughput).
    # Measure over ~2 seconds to get the true number.
    print("[CAM] Measuring actual FPS...")
    fps_frames = 0
    fps_t0 = time.time()
    while time.time() - fps_t0 < 2.0:
        ret, _ = cap.read()
        if ret: fps_frames += 1
    measured_fps = fps_frames / (time.time() - fps_t0)

    # Store on the capture object so scripts can read it
    cap._measured_fps = round(measured_fps, 1)
    print(f"[CAM] Ready. Actual throughput: {cap._measured_fps} FPS")
    return cap


def get_camera_fps(cap, sitl=False):
    """Get the real FPS for a camera. Use this for VideoWriter."""
    if hasattr(cap, '_measured_fps') and cap._measured_fps > 1:
        return cap._measured_fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 1:
        return fps
    return 30 if sitl else 21  # safe fallback


# ===========================================================================
# YOLO
# ===========================================================================
_yolo_model = None

def load_yolo(weights=DEFAULT_WEIGHTS, imgsz=DEFAULT_IMGSZ):
    global _yolo_model
    if _yolo_model: return _yolo_model
    from ultralytics import YOLO
    import numpy as np

    for p in [weights, os.path.expanduser(f"~/{weights}")]:
        if os.path.exists(p): weights = p; break
    else:
        print(f"[!] Model not found: {weights}"); sys.exit(1)
    print(f"[YOLO] Loading {weights}...")
    _yolo_model = YOLO(weights)
    print(f"[YOLO] Classes: {_yolo_model.names}")

    # ── Device check ──
    try:
        device = next(_yolo_model.model.parameters()).device
        print(f"[YOLO] Device: {device}")
        if str(device) == "cpu":
            print("[YOLO] ⚠ Running on CPU — inference will be SLOW (2-3s/frame)")
            print("[YOLO]   Check: python3 -c \"import torch; print(torch.cuda.is_available())\"")
    except Exception:
        print("[YOLO] Device: unknown")

    # ── Warmup inference ──
    # First YOLO call is always slow (CUDA kernel compilation, memory alloc).
    # Do it here so it doesn't eat into flight time.
    print("[YOLO] Warmup inference...")
    t0 = time.time()
    _dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = _yolo_model(_dummy, imgsz=imgsz, conf=0.5, verbose=False)
    warmup_ms = (time.time() - t0) * 1000
    print(f"[YOLO] Warmup done: {warmup_ms:.0f}ms")
    if warmup_ms > 1000:
        print(f"[YOLO] ⚠ Inference is slow ({warmup_ms:.0f}ms) — likely on CPU!")

    return _yolo_model

def detect_x(frame, model=None, conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ):
    if model is None: model = load_yolo()
    results = model(frame, imgsz=imgsz, conf=conf, verbose=False)
    best, best_c = None, 0.0
    for r in results:
        for box in r.boxes:
            c = float(box.conf[0])
            if c >= conf and c > best_c:
                best_c = c
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                best = {"conf":c, "cx":(x1+x2)//2, "cy":(y1+y2)//2,
                        "bbox":(x1,y1,x2,y2), "class_name":r.names[int(box.cls[0])]}
    return best


# ===========================================================================
# PIXEL → METERS
# ===========================================================================
def pixels_to_meters(dx_px, dy_px, alt_m):
    """Camera down. Image top = drone forward. Returns (fwd_m, right_m)."""
    hfov = math.radians(HFOV_DEG)
    vfov = hfov * (FRAME_H / FRAME_W)
    gw = 2 * alt_m * math.tan(hfov/2)
    gh = 2 * alt_m * math.tan(vfov/2)
    return -dy_px * (gh/FRAME_H), dx_px * (gw/FRAME_W)


# ===========================================================================
# SAFE FLIGHT
# ===========================================================================
class SafeFlight:
    def __init__(self, fc, camera=None, video_writer=None):
        self.fc = fc; self.cam = camera; self.vw = video_writer; self._osig = None
    def __enter__(self):
        self._osig = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._emer)
        signal.signal(signal.SIGTERM, self._emer)
        return self
    def __exit__(self, et, ev, tb):
        signal.signal(signal.SIGINT, self._osig or signal.SIG_DFL)
        if et:
            print(f"\n[!!!] {et.__name__}: {ev}\n[!!!] EMERGENCY RTL!")
            try: self.fc.set_rtl()
            except: pass
        self._clean()
        return False
    def _emer(self, s, f):
        print("\n[!!!] Ctrl+C → EMERGENCY RTL!")
        try: self.fc.set_rtl()
        except: pass
        self._clean(); sys.exit(0)
    def _clean(self):
        if self.vw:
            try: self.vw.release(); print("[*] Video saved.")
            except: pass
        if self.cam:
            try: self.cam.release()
            except: pass
        try: self.fc.close()
        except: pass


# ===========================================================================
# LOGGING
# ===========================================================================
def create_log(prefix="flight"):
    fname = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    f = open(fname, 'w')
    f.write(f"# {prefix} — {datetime.now().isoformat()}\n\n")
    print(f"[LOG] → {fname}")
    return fname, f

def log(f, msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    print(line)
    if f: f.write(line + "\n"); f.flush()


# ===========================================================================
# CONFIRMATION
# ===========================================================================
def confirm(name, desc):
    print("\n" + "!" * 60)
    print(f"  {name}")
    print(f"  {desc}")
    print("!" * 60)
    print("\n  CHECKLIST:")
    print("  [ ] MAVProxy running in another terminal")
    print("  [ ] Battery >50%  |  Area clear  |  GPS lock")
    print("  [ ] Ready to type 'mode RTL' in MAVProxy if needed")
    print()
    if input("  Type 'yes' to fly: ").strip().lower() != 'yes':
        print("[*] Aborted."); return False
    print(); return True
