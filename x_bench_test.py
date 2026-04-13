#!/usr/bin/env python3
"""
X Landing — Phase 2 Bench Test (v2 — DO_MOTOR_TEST)
=====================================================
Fixed camera + handheld X target + Pixhawk connected (PROPS OFF!)

WHY v2:
  The v1 script used GUIDED mode + SET_POSITION_TARGET_LOCAL_NED.
  That requires GPS lock + EKF + pre-arm checks → fails on the bench.
  
  This version uses DO_MOTOR_TEST (proven working on your setup)
  and maps the X offset to DIFFERENTIAL motor throttle so you can
  verify the correct motors respond to the correct direction.

How it works:
  - Detects X, computes offset from frame center
  - Converts offset to motor throttle per motor (1-4)
  - Motors on the "correction side" spin faster
  - Example: X is LEFT of center → drone needs to go LEFT
    → motors on right side spin faster to tilt left
  
  MOTOR LAYOUT (ArduCopter default X-frame, looking from top):
  
       Front
    1 (CW)   2 (CCW)
        \\ /
         X
        / \\
    4 (CCW)  3 (CW)
       Back
  
  To move LEFT  → tilt left  → right motors (2,3) spin faster
  To move RIGHT → tilt right → left motors (1,4) spin faster
  To move FWD   → tilt fwd   → back motors (3,4) spin faster
  To move BACK  → tilt back  → front motors (1,2) spin faster

  *** YOUR FRAME MAY BE DIFFERENT ***
  Run the test, see which motors spin, flip the mapping if wrong.
  That's the whole point of Phase 2!

Setup:
  1. Mount camera pointing down at table/floor
  2. Connect Pixhawk via USB-C (/dev/ttyACM0)
  3. REMOVE ALL PROPS!!!
  4. Hold hardboard X target, move it around
  5. Watch which motors respond — verify direction is correct

Usage:
    python3 x_bench_test.py --dry-run            # logic only, no Pixhawk
    python3 x_bench_test.py                      # live motor test
    python3 x_bench_test.py --log                # + save CSV
    python3 x_bench_test.py --base-throttle 10   # gentler (default 15%)
    python3 x_bench_test.py --headless           # SSH, no display window

REMOVE PROPS FOR TESTING!

Transfer:
    scp x_bench_test.py best_22.pt jetson@<ip>:~/
"""

import argparse
import time
import csv
import cv2
import os
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
    dx = cx - mid_x  # positive = X is RIGHT of center
    dy = cy - mid_y  # positive = X is BELOW center

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
# Convert offset to per-motor throttle
# ---------------------------------------------------------------------------
def offset_to_motor_throttle(dx, dy, frame_w, frame_h,
                              base_throttle=15, max_diff=10, deadzone=50):
    """
    Convert pixel offset to throttle for motors 1-4.

    ArduCopter X-frame (default, top view):

         Front
      M1(CW)    M2(CCW)
          \\      /
            X
          /      \\
      M4(CCW)   M3(CW)
         Back

    To move in a direction, the OPPOSITE side motors spin faster:
      Move LEFT  → right side (M2, M3) spin more
      Move RIGHT → left side  (M1, M4) spin more
      Move FWD   → back side  (M3, M4) spin more
      Move BACK  → front side (M1, M2) spin more

    Returns: (m1, m2, m3, m4) throttle percentages
    
    *** IF MOTORS RESPOND WRONG, FLIP SIGNS BELOW ***
    """
    # Normalize to [-1, 1]
    norm_x = dx / (frame_w / 2)
    norm_y = dy / (frame_h / 2)

    # Apply deadzone
    dz_x = deadzone / (frame_w / 2)
    dz_y = deadzone / (frame_h / 2)
    if abs(norm_x) < dz_x:
        norm_x = 0.0
    if abs(norm_y) < dz_y:
        norm_y = 0.0

    # Clamp
    norm_x = max(-1.0, min(1.0, norm_x))
    norm_y = max(-1.0, min(1.0, norm_y))

    # Differential throttle
    roll_diff = norm_x * max_diff    # positive = X is right → need to go right
    pitch_diff = norm_y * max_diff   # positive = X is below → need to go back

    # === MOTOR MIXING (edit here if wrong direction!) ===
    # Each motor gets: base + roll_contribution + pitch_contribution
    #
    # To go RIGHT: left motors (M1, M4) spin faster → +roll_diff on M1,M4
    # To go BACK:  front motors (M1, M2) spin faster → +pitch_diff on M1,M2

    m1 = base_throttle + roll_diff + pitch_diff   # front-left
    m2 = base_throttle - roll_diff + pitch_diff   # front-right
    m3 = base_throttle - roll_diff - pitch_diff   # back-right
    m4 = base_throttle + roll_diff - pitch_diff   # back-left

    # Clamp to [0, base_throttle + max_diff]
    max_t = base_throttle + max_diff
    m1 = max(0, min(max_t, m1))
    m2 = max(0, min(max_t, m2))
    m3 = max(0, min(max_t, m3))
    m4 = max(0, min(max_t, m4))

    return m1, m2, m3, m4


# ---------------------------------------------------------------------------
# Pixhawk connection + motor commands (from your working x_detect_motor.py)
# ---------------------------------------------------------------------------
def connect_pixhawk(device, baud):
    from pymavlink import mavutil

    print(f"[MAV] Connecting to {device} at {baud}...")
    master = mavutil.mavlink_connection(device, baud=baud)
    print("[MAV] Waiting for heartbeat...")
    master.wait_heartbeat(timeout=30)
    master.target_component = 1
    print(f"[MAV] Connected! system={master.target_system}")
    return master


def wait_cmd_ack(master, command_id, timeout=3):
    start = time.time()
    while time.time() - start < timeout:
        try:
            msg = master.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.5)
            if msg and msg.command == command_id:
                return msg
        except Exception:
            pass
    return None


def force_arm(master):
    """Force arm using 21196 — same as your working x_detect_motor.py."""
    from pymavlink import mavutil

    print("[MAV] >>> FORCE ARMING <<<")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 21196, 0, 0, 0, 0, 0)

    ack = wait_cmd_ack(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
    if ack and ack.result == 0:
        print("[MAV] Armed!")
        return True
    else:
        print(f"[MAV] Arm result: {ack.result if ack else 'no response'}")
        return False


def force_disarm(master):
    from pymavlink import mavutil

    print("[MAV] >>> FORCE DISARMING <<<")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, 21196, 0, 0, 0, 0, 0)

    ack = wait_cmd_ack(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
    if ack and ack.result == 0:
        print("[MAV] Disarmed!")
    else:
        print(f"[MAV] Disarm result: {ack.result if ack else 'no response'}")


def send_motor_test(master, motor_num, throttle_pct, duration=1.0):
    """
    Send DO_MOTOR_TEST — same method as your working x_detect_motor.py.
    motor_num: 1-4
    throttle_pct: 0-100
    duration: seconds (use short duration, we resend each loop)
    """
    from pymavlink import mavutil

    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST,
        0,
        motor_num,       # motor instance (1-4)
        0,               # throttle type: 0 = percent
        throttle_pct,    # throttle value
        duration,        # timeout in seconds
        1,               # motor count
        0, 0
    )


def send_all_motors(master, m1, m2, m3, m4, duration=1.5):
    """Send throttle to all 4 motors. Duration slightly > loop interval."""
    send_motor_test(master, 1, m1, duration)
    send_motor_test(master, 2, m2, duration)
    send_motor_test(master, 3, m3, duration)
    send_motor_test(master, 4, m4, duration)


def stop_all_motors(master):
    """Send zero throttle to all motors."""
    for m in [1, 2, 3, 4]:
        send_motor_test(master, m, 0, 0.5)


# ---------------------------------------------------------------------------
# CSV Logger
# ---------------------------------------------------------------------------
class BenchLogger:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.file = None
        self.writer = None
        if enabled:
            fname = f"bench_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.file = open(fname, 'w', newline='')
            self.writer = csv.writer(self.file)
            self.writer.writerow([
                'timestamp', 'frame_num',
                'x_detected', 'conf',
                'cx', 'cy',
                'dx_px', 'dy_px', 'centered',
                'm1_throttle', 'm2_throttle', 'm3_throttle', 'm4_throttle',
                'direction', 'fps'
            ])
            print(f"[LOG] Logging to: {fname}")

    def log(self, frame_num, detected, conf, cx, cy,
            dx, dy, centered, m1, m2, m3, m4, direction, fps):
        if not self.enabled:
            return
        self.writer.writerow([
            datetime.now().isoformat(), frame_num,
            detected, f"{conf:.3f}",
            cx, cy, dx, dy, centered,
            f"{m1:.1f}", f"{m2:.1f}", f"{m3:.1f}", f"{m4:.1f}",
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
    print("  X LANDING — PHASE 2 BENCH TEST (DO_MOTOR_TEST)")
    print("=" * 65)
    if args.dry_run:
        print("  MODE: DRY RUN (no Pixhawk, logic only)")
    else:
        print(f"  MODE: LIVE (Pixhawk @ {args.device})")
    print(f"  BASE THROTTLE: {args.base_throttle}%")
    print(f"  MAX DIFFERENTIAL: +/- {args.max_diff}%")
    print(f"  DEADZONE: {args.deadzone}px")
    print(f"  CENTER METHOD: {'refined (green seg)' if args.refine else 'bbox midpoint'}")
    print("=" * 65)

    # Load model
    print("\n[*] Loading YOLO model...")
    model = load_model(args.weights)

    # Connect Pixhawk
    master = None
    armed = False
    if not args.dry_run:
        master = connect_pixhawk(args.device, args.baud)
        armed = force_arm(master)
        if not armed:
            print("[!] Failed to arm — motors won't spin.")
            print("    Check: is Pixhawk powered properly via USB-C?")
            print("    Continuing to show detection logic...")

    # Open camera
    mode = MODES[args.mode]
    pipeline = build_gstreamer_pipeline(**mode)
    infer_w = mode["display_width"]
    infer_h = mode["display_height"]
    print(f"[*] Opening camera: {args.mode} → {infer_w}x{infer_h}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[!] Failed to open camera.")
        if armed and not args.dry_run:
            force_disarm(master)
        return

    # Logger
    logger = BenchLogger(args.log)

    # Display setup
    view_w, view_h = 960, 540
    show_display = not args.headless
    if show_display:
        window_name = "X Landing — Bench Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, view_w, view_h)

    # Loop state
    fps_count = 0
    fps_start = time.time()
    display_fps = 0.0
    frame_num = 0
    last_cmd_time = 0
    cmd_interval = 0.5  # send motor commands every 0.5s

    # Track if motors are currently spinning
    motors_active = False
    last_detection_time = 0
    no_detect_timeout = 2.0  # stop motors after 2s without X

    method_tag = "REFINED" if args.refine else "BBOX"
    print(f"\n[*] Running bench test. Move your X target around!")
    print(f"    Press 'q' to quit, 's' for snapshot")
    print(f"    Motors spin when X detected, stop when X lost for {no_detect_timeout}s")
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

            # --- Compute offset + motor throttle ---
            cx, cy = 0, 0
            dx, dy = 0, 0
            centered = False
            direction_str = "NO TARGET"
            m1, m2, m3, m4 = 0, 0, 0, 0

            if best_box is not None:
                last_detection_time = now

                if args.refine:
                    cx, cy = find_x_center_refined(frame, best_box)
                else:
                    cx, cy = find_x_center_bbox(best_box)

                dx, dy, centered, direction_str = compute_offset(
                    cx, cy, frame_w, frame_h, args.deadzone
                )

                if centered:
                    # All motors at base (hovering steady)
                    m1 = m2 = m3 = m4 = args.base_throttle
                else:
                    # Differential throttle based on offset
                    m1, m2, m3, m4 = offset_to_motor_throttle(
                        dx, dy, frame_w, frame_h,
                        base_throttle=args.base_throttle,
                        max_diff=args.max_diff,
                        deadzone=args.deadzone
                    )

                # Send motor commands
                if not args.dry_run and master and armed:
                    if now - last_cmd_time >= cmd_interval:
                        send_all_motors(master, m1, m2, m3, m4, duration=1.5)
                        last_cmd_time = now
                        motors_active = True

            else:
                # No detection — stop motors after timeout
                if motors_active and (now - last_detection_time > no_detect_timeout):
                    if not args.dry_run and master and armed:
                        stop_all_motors(master)
                        motors_active = False
                    direction_str = "NO TARGET — motors stopped"

            # --- Log ---
            logger.log(
                frame_num, det_count > 0, best_conf,
                cx, cy, dx, dy, centered,
                m1, m2, m3, m4, direction_str, display_fps
            )

            # --- Terminal output ---
            if det_count > 0:
                motor_str = f"M[{m1:.0f},{m2:.0f},{m3:.0f},{m4:.0f}]%"
                if centered:
                    print(f"\r[CENTERED] conf={best_conf:.2f} "
                          f"X@({cx},{cy}) {motor_str} "
                          f"FPS={display_fps:.1f}           ", end="", flush=True)
                else:
                    print(f"\r[{direction_str}] conf={best_conf:.2f} "
                          f"{motor_str} "
                          f"FPS={display_fps:.1f}    ", end="", flush=True)
            else:
                print(f"\rSearching... FPS={display_fps:.1f} | No X    ",
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

                # Draw all detections
                for result in results:
                    for box in result.boxes:
                        conf_val = float(box.conf[0])
                        if conf_val < args.conf:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_name = result.names[int(box.cls[0])]
                        dx1 = int(x1 * scale_x)
                        dy1 = int(y1 * scale_y)
                        dx2 = int(x2 * scale_x)
                        dy2 = int(y2 * scale_y)
                        cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2),
                                      (0, 255, 0), 2)
                        label = f"{cls_name} {conf_val:.2f}"
                        lsz, _ = cv2.getTextSize(label,
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(display_frame, (dx1, dy1 - lsz[1] - 10),
                                      (dx1 + lsz[0], dy1), (0, 255, 0), -1)
                        cv2.putText(display_frame, label, (dx1, dy1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Draw best detection center + guidance
                if best_box is not None:
                    dcx = int(cx * scale_x)
                    dcy = int(cy * scale_y)
                    dot_color = (0, 255, 0) if centered else (0, 0, 255)
                    cv2.circle(display_frame, (dcx, dcy), 8, dot_color, -1)
                    cv2.circle(display_frame, (dcx, dcy), 10, (255, 255, 255), 2)
                    cv2.line(display_frame, (dcx, dcy),
                             (smid_x, smid_y), dot_color, 2)

                # Motor throttle bars (bottom-right corner)
                bar_x = view_w - 180
                bar_y = view_h - 130
                motor_vals = [m1, m2, m3, m4]
                motor_labels = ["M1(FL)", "M2(FR)", "M3(BR)", "M4(BL)"]
                max_t = args.base_throttle + args.max_diff
                for i, (val, lbl) in enumerate(zip(motor_vals, motor_labels)):
                    y_pos = bar_y + i * 28
                    bar_len = int((val / max(max_t, 1)) * 120)
                    color = (0, 200, 255) if val > 0 else (80, 80, 80)
                    cv2.rectangle(display_frame,
                                  (bar_x, y_pos), (bar_x + bar_len, y_pos + 20),
                                  color, -1)
                    cv2.rectangle(display_frame,
                                  (bar_x, y_pos), (bar_x + 120, y_pos + 20),
                                  (150, 150, 150), 1)
                    cv2.putText(display_frame, f"{lbl}:{val:.0f}%",
                                (bar_x - 95, y_pos + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                # Guidance text
                if centered:
                    guide_text = "*** CENTERED — ALL EQUAL ***"
                    guide_color = (0, 255, 0)
                elif det_count > 0:
                    guide_text = direction_str
                    guide_color = (0, 165, 255)
                else:
                    guide_text = "Searching..."
                    guide_color = (150, 150, 150)

                cv2.putText(display_frame, guide_text, (10, view_h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, guide_color, 2)

                # Offset readout
                if det_count > 0:
                    off_text = f"offset: ({dx:+d}, {dy:+d})px"
                    cv2.putText(display_frame, off_text, (10, view_h - 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Status bar
                mode_str = "DRY RUN" if args.dry_run else ("ARMED" if armed else "NOT ARMED")
                active_str = "MOTORS ON" if motors_active else "STANDBY"
                info = (f"FPS:{display_fps:.1f} | X:{det_count} | "
                        f"{method_tag} | {mode_str} | {active_str}")
                bar_color = (200, 200, 200) if args.dry_run else (0, 200, 255)
                if motors_active:
                    bar_color = (0, 0, 255)
                    cv2.rectangle(display_frame, (0, 0),
                                  (view_w - 1, view_h - 1), (0, 0, 255), 3)
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
        # Safety: always stop motors and disarm
        if not args.dry_run and master:
            print("\n[*] Stopping motors...")
            stop_all_motors(master)
            time.sleep(0.5)
            force_disarm(master)

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
        description="X Landing — Phase 2 Bench Test (PROPS OFF!)")

    # Detection
    parser.add_argument("--weights", default="best_22.pt",
                        help="YOLO weights (default: best_22.pt)")
    parser.add_argument("--mode", choices=["4k", "1080p"], default="1080p",
                        help="Camera mode (default: 1080p)")
    parser.add_argument("--conf", type=float, default=0.50,
                        help="Confidence threshold (default: 0.50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO inference size (default: 640)")
    parser.add_argument("--refine", action="store_true",
                        help="Use green segmentation for precise X center")
    parser.add_argument("--deadzone", type=int, default=50,
                        help="Pixel deadzone for 'centered' (default: 50)")

    # Pixhawk
    parser.add_argument("--device", default="/dev/ttyACM0",
                        help="Pixhawk serial port (default: /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200,
                        help="Baud rate (default: 115200)")
    parser.add_argument("--base-throttle", type=int, default=15,
                        help="Base motor throttle percent (default: 15)")
    parser.add_argument("--max-diff", type=int, default=10,
                        help="Max differential throttle +/- percent (default: 10)")

    # Modes
    parser.add_argument("--dry-run", action="store_true",
                        help="No Pixhawk — detection + offset calc only")
    parser.add_argument("--headless", action="store_true",
                        help="No display window (SSH)")
    parser.add_argument("--log", action="store_true",
                        help="Save all data to CSV file")

    args = parser.parse_args()

    if not args.dry_run:
        print("\n" + "!" * 65)
        print("  WARNING: This script SPINS MOTORS using DO_MOTOR_TEST!")
        print("  REMOVE ALL PROPS before continuing.")
        print("!" * 65)
        resp = input("\n  Props removed? (y/n): ").strip().lower()
        if resp != 'y':
            print("[*] Aborted.")
            exit(0)

    run_bench(args)
