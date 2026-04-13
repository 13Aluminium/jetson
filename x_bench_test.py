#!/usr/bin/env python3
"""
X Landing — Phase 2 Bench Test
================================
Fixed camera + handheld X target + Pixhawk connected (PROPS OFF!)

What this tests:
  1. YOLO detects the X
  2. Centering offset is computed (from x_detect_guide.py logic)
  3. Offset is converted to velocity commands
  4. Commands are sent to Pixhawk via MAVLink (GUIDED mode)
  5. You verify the RIGHT motors respond to the RIGHT direction

Setup:
  - Mount camera pointing down (or at an angle — doesn't matter for bench)
  - Connect Pixhawk via USB (/dev/ttyACM0)
  - REMOVE ALL PROPS
  - Hold your hardboard X target in front of the camera
  - Move it around and watch which motors respond

Modes:
  --dry-run       : detection + offset calc only, no Pixhawk (Phase 1 test)
  --log           : save all offsets + commands to CSV for review
  (default)       : full Phase 2 — sends commands to Pixhawk

Usage:
    python3 x_bench_test.py --dry-run              # Phase 1: logic only
    python3 x_bench_test.py                        # Phase 2: Pixhawk connected
    python3 x_bench_test.py --log                  # Phase 2 + CSV logging
    python3 x_bench_test.py --throttle 15          # gentler motor test
    python3 x_bench_test.py --headless             # SSH, no display

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
# Compute offset and direction
# ---------------------------------------------------------------------------
def compute_offset(cx, cy, frame_w, frame_h, deadzone=50):
    """
    Returns (dx, dy, centered, direction_str)
    dx > 0 means X is RIGHT of center  → drone should move RIGHT
    dy > 0 means X is BELOW center     → drone should move FORWARD
    (assuming camera points straight down, forward = +Y in camera)
    """
    mid_x = frame_w // 2
    mid_y = frame_h // 2
    dx = cx - mid_x
    dy = cy - mid_y

    if abs(dx) <= deadzone and abs(dy) <= deadzone:
        return dx, dy, True, "CENTERED"

    parts = []
    if dx < -deadzone:
        parts.append(f"LEFT ({abs(dx)}px)")
    elif dx > deadzone:
        parts.append(f"RIGHT ({abs(dx)}px)")

    if dy < -deadzone:
        parts.append(f"UP ({abs(dy)}px)")
    elif dy > deadzone:
        parts.append(f"DOWN ({abs(dy)}px)")

    return dx, dy, False, " + ".join(parts)


# ---------------------------------------------------------------------------
# Convert pixel offset to velocity command
# ---------------------------------------------------------------------------
def pixel_offset_to_velocity(dx, dy, frame_w, frame_h, max_speed=0.5, deadzone=50):
    """
    Convert pixel offset to NED velocity (m/s).

    Camera-to-NED mapping (camera pointing straight down):
      Camera +X (right)  → NED +East  (vy)
      Camera +Y (down)   → NED +North (vx)  [forward in body frame]

    NOTE: This mapping depends on how your camera is mounted.
    If motors respond in wrong direction during bench test,
    flip the signs here. That's the whole point of Phase 2!

    Speed is proportional to offset, capped at max_speed.
    Inside deadzone → zero velocity.
    """
    # normalize offset to [-1, 1]
    norm_x = dx / (frame_w / 2)
    norm_y = dy / (frame_h / 2)

    # apply deadzone
    dz_norm_x = deadzone / (frame_w / 2)
    dz_norm_y = deadzone / (frame_h / 2)

    if abs(norm_x) < dz_norm_x:
        norm_x = 0
    if abs(norm_y) < dz_norm_y:
        norm_y = 0

    # clamp to [-1, 1]
    norm_x = max(-1.0, min(1.0, norm_x))
    norm_y = max(-1.0, min(1.0, norm_y))

    # --- CAMERA TO NED MAPPING ---
    # Edit these if bench test shows wrong motor direction!
    vx = norm_y * max_speed   # camera Y → NED north (forward)
    vy = norm_x * max_speed   # camera X → NED east  (right)
    vz = 0.0                  # no vertical for bench test

    return vx, vy, vz


# ---------------------------------------------------------------------------
# Pixhawk connection
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


def set_guided_mode(master):
    """Switch to GUIDED mode (mode number 4 for copter)."""
    from pymavlink import mavutil

    print("[MAV] Setting GUIDED mode...")
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        4  # GUIDED mode for ArduCopter
    )
    time.sleep(1)
    # verify
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
        0, 0, 0, 0, 0, 0, 0, 0)
    print("[MAV] GUIDED mode set")


def force_arm(master):
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


def send_velocity_ned(master, vx, vy, vz):
    """
    Send velocity command in NED frame.
    vx = north (forward), vy = east (right), vz = down (positive = descend)
    """
    from pymavlink import mavutil

    # type_mask: ignore position, acceleration, yaw — only velocity
    type_mask = (
        0b0000_1111_11_000_111  # bits: pos=ignore, vel=use, accel=ignore, yaw=ignore
    )
    # More readable: ignore everything except vx, vy, vz
    type_mask = 0b0000_1100_11_000_111  # = 0x0C07... let me be explicit:
    # bit 0: ignore px        = 1
    # bit 1: ignore py        = 1
    # bit 2: ignore pz        = 1
    # bit 3: ignore vx        = 0  (USE)
    # bit 4: ignore vy        = 0  (USE)
    # bit 5: ignore vz        = 0  (USE)
    # bit 6: ignore afx       = 1
    # bit 7: ignore afy       = 1
    # bit 8: ignore afz       = 1
    # bit 9: ignore yaw       = 1
    # bit 10: ignore yaw_rate = 1
    type_mask = 0b0000_11_111_000_111  # = 0x07C7
    # Let me just compute it properly:
    type_mask = (
        (1 << 0) |  # ignore px
        (1 << 1) |  # ignore py
        (1 << 2) |  # ignore pz
        # bit 3,4,5 = 0 → USE vx, vy, vz
        (1 << 6) |  # ignore afx
        (1 << 7) |  # ignore afy
        (1 << 8) |  # ignore afz
        (1 << 9) |  # ignore yaw
        (1 << 10)   # ignore yaw_rate
    )  # = 0x07C7 = 1991

    master.mav.set_position_target_local_ned_send(
        0,                                          # time_boot_ms (ignored)
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
            fname = f"bench_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.file = open(fname, 'w', newline='')
            self.writer = csv.writer(self.file)
            self.writer.writerow([
                'timestamp', 'frame_num',
                'x_detected', 'conf',
                'cx', 'cy',
                'dx_px', 'dy_px', 'centered',
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
    print("  X LANDING — PHASE 2 BENCH TEST")
    print("=" * 65)
    if args.dry_run:
        print("  MODE: DRY RUN (no Pixhawk, logic only)")
    else:
        print(f"  MODE: LIVE (Pixhawk @ {args.device})")
        print(f"  THROTTLE CAP: max_speed = {args.max_speed} m/s")
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
        from pymavlink import mavutil
        master = connect_pixhawk(args.device, args.baud)
        set_guided_mode(master)
        armed = force_arm(master)
        if not armed:
            print("[!] Failed to arm. Continuing anyway for bench test...")

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
    cmd_interval = 0.1  # send commands at 10 Hz max

    # Safety: timeout with no detection → disarm
    last_detection_time = time.time()
    no_detect_timeout = 5.0  # seconds

    method_tag = "REFINED" if args.refine else "BBOX"
    print(f"\n[*] Running bench test. Move your X target around!")
    print(f"    Press 'q' to quit, 's' for snapshot")
    print(f"    If no X for {no_detect_timeout}s → velocity zeroed")
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
                # else: velocity stays zero = hold position

            else:
                # No detection
                if now - last_detection_time > no_detect_timeout:
                    direction_str = "NO TARGET (timeout)"

            # --- Send velocity command to Pixhawk ---
            if not args.dry_run and master and (now - last_cmd_time >= cmd_interval):
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
                if centered:
                    print(f"\r[CENTERED] conf={best_conf:.2f} "
                          f"X@({cx},{cy}) offset=({dx:+d},{dy:+d}) "
                          f"vel=(0,0,0) "
                          f"FPS={display_fps:.1f}           ", end="", flush=True)
                else:
                    print(f"\r[{direction_str}] conf={best_conf:.2f} "
                          f"X@({cx},{cy}) offset=({dx:+d},{dy:+d}) "
                          f"vel=({vx:+.2f},{vy:+.2f},{vz:+.2f}) "
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
                        conf = float(box.conf[0])
                        if conf < args.conf:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_name = result.names[int(box.cls[0])]
                        dx1 = int(x1 * scale_x)
                        dy1 = int(y1 * scale_y)
                        dx2 = int(x2 * scale_x)
                        dy2 = int(y2 * scale_y)
                        cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2),
                                      (0, 255, 0), 2)
                        label = f"{cls_name} {conf:.2f}"
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

                    # Velocity vector visualization (scaled up for visibility)
                    arrow_scale = 200
                    arrow_end_x = smid_x + int(vy * arrow_scale)  # vy = east = right
                    arrow_end_y = smid_y + int(vx * arrow_scale)  # vx = north = up... 
                    # but on screen, down is +Y, so north (forward) = up on screen
                    arrow_end_y = smid_y - int(vx * arrow_scale)
                    cv2.arrowedLine(display_frame, (smid_x, smid_y),
                                    (arrow_end_x, arrow_end_y),
                                    (0, 255, 255), 3, tipLength=0.3)

                # Guidance text
                if centered:
                    guide_text = "*** CENTERED — HOLD ***"
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
                vel_text = f"vel NED: ({vx:+.2f}, {vy:+.2f}, {vz:+.2f}) m/s"
                cv2.putText(display_frame, vel_text, (10, view_h - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Offset readout
                if det_count > 0:
                    off_text = f"offset: ({dx:+d}, {dy:+d})px"
                    cv2.putText(display_frame, off_text, (10, view_h - 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Status bar
                mode_str = "DRY RUN" if args.dry_run else "PIXHAWK LIVE"
                info = (f"FPS:{display_fps:.1f} | X:{det_count} | "
                        f"{method_tag} | {mode_str}")
                bar_color = (200, 200, 200) if args.dry_run else (0, 200, 255)
                cv2.putText(display_frame, info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, bar_color, 2)

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
        # Safety: always zero velocity and disarm
        if not args.dry_run and master:
            print("\n[*] Zeroing velocity...")
            send_velocity_ned(master, 0, 0, 0)
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
    parser.add_argument("--max-speed", type=float, default=0.3,
                        help="Max velocity m/s for bench test (default: 0.3)")

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
        print("  WARNING: This script sends commands to Pixhawk!")
        print("  REMOVE ALL PROPS before continuing.")
        print("!" * 65)
        resp = input("\n  Props removed? (y/n): ").strip().lower()
        if resp != 'y':
            print("[*] Aborted.")
            exit(0)

    run_bench(args)
