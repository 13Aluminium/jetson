#!/usr/bin/env python3
"""
X Target Detector + Motor Control — Jetson Orin Nano + IMX477 + Pixhawk 6C
===========================================================================
- Detects X using YOLO (your custom best_22.pt)
- Shows live camera feed on monitor with bounding boxes
- When X is detected: arms Pixhawk and spins ALL motors at 100% for 10 seconds
- After 10 seconds: stops motors and disarms
- Uses pymavlink (the method that actually works on your setup)

Usage:
    python3 x_detect_motor.py                          # live view + motor control
    python3 x_detect_motor.py --dry-run                # live view only, no motors
    python3 x_detect_motor.py --conf 0.6               # higher confidence
    python3 x_detect_motor.py --device /dev/ttyACM1    # different serial port
    python3 x_detect_motor.py --throttle 50            # 50% instead of 100%

REMOVE PROPS FOR TESTING!
"""

import argparse
import time
import cv2
import os
from datetime import datetime
from pymavlink import mavutil


# ---------------------------------------------------------------------------
# GStreamer pipeline (from your working script)
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
    print(f"[+] Model loaded: {weights}")
    print(f"    Classes: {model.names}")
    return model


# ---------------------------------------------------------------------------
# Connect to Pixhawk
# ---------------------------------------------------------------------------
def connect_pixhawk(device, baud):
    print(f"[MAV] Connecting to {device} at {baud}...")
    master = mavutil.mavlink_connection(device, baud=baud)
    print("[MAV] Waiting for heartbeat...")
    master.wait_heartbeat(timeout=30)
    master.target_component = 1
    print(f"[MAV] Connected! system={master.target_system}")
    return master


# ---------------------------------------------------------------------------
# Wait for command ACK
# ---------------------------------------------------------------------------
def wait_cmd_ack(master, command_id, timeout=5):
    start = time.time()
    while time.time() - start < timeout:
        try:
            msg = master.recv_match(type="COMMAND_ACK", blocking=True, timeout=1)
            if msg and msg.command == command_id:
                return msg
        except:
            pass
    return None


# ---------------------------------------------------------------------------
# Force arm + spin all motors
# ---------------------------------------------------------------------------
def arm_and_spin_motors(master, throttle_pct=100, duration=10):
    """
    Force arm the Pixhawk, then spin all 4 motors using DO_MOTOR_TEST.
    throttle_pct: 0-100
    duration: seconds to spin
    """
    # Force arm (21196 bypasses all checks)
    print(f"[MAV] >>> FORCE ARMING <<<")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 21196, 0, 0, 0, 0, 0)

    ack = wait_cmd_ack(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
    if ack and ack.result == 0:
        print("[MAV] Armed!")
    else:
        print(f"[MAV] Arm ACK: {ack.result if ack else 'no response'}")

    time.sleep(0.5)

    # Spin all 4 motors using DO_MOTOR_TEST
    print(f"[MAV] Spinning ALL motors at {throttle_pct}% for {duration}s...")
    for motor in [1, 2, 3, 4]:
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST,
            0,
            motor,          # motor instance
            0,              # throttle type: 0 = percent
            throttle_pct,   # throttle value
            duration,       # duration in seconds
            1,              # motor count (unused but required)
            0, 0)
        ack = wait_cmd_ack(master, mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST)
        result = ack.result if ack else "no response"
        print(f"    Motor {motor}: ACK={result}")

    return True


# ---------------------------------------------------------------------------
# Force disarm
# ---------------------------------------------------------------------------
def force_disarm(master):
    print("[MAV] >>> FORCE DISARMING <<<")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, 21196, 0, 0, 0, 0, 0)
    ack = wait_cmd_ack(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
    if ack and ack.result == 0:
        print("[MAV] Disarmed!")
    else:
        print(f"[MAV] Disarm ACK: {ack.result if ack else 'no response'}")


# ---------------------------------------------------------------------------
# Draw detections (from your working script)
# ---------------------------------------------------------------------------
def draw_detections(frame, results, conf_thresh=0.50, scale_x=1.0, scale_y=1.0):
    detections = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append({
                "class": cls_name,
                "conf": conf,
                "center": (cx, cy),
                "bbox": (x1, y1, x2, y2),
            })

            # Draw on display frame
            dx1 = int(x1 * scale_x)
            dy1 = int(y1 * scale_y)
            dx2 = int(x2 * scale_x)
            dy2 = int(y2 * scale_y)

            cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (dx1, dy1 - label_size[1] - 10),
                          (dx1 + label_size[0], dy1), (0, 255, 0), -1)
            cv2.putText(frame, label, (dx1, dy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame, detections


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(args):
    # Load YOLO model
    print("[*] Loading X detection model...")
    model = load_model(args.weights)

    # Connect to Pixhawk
    master = None
    if not args.dry_run:
        master = connect_pixhawk(args.device, args.baud)
    else:
        print("[DRY RUN] Skipping Pixhawk connection")

    # Open camera
    mode = MODES[args.mode]
    pipeline = build_gstreamer_pipeline(**mode)
    print(f"[*] Opening camera: {args.mode} → "
          f"{mode['display_width']}x{mode['display_height']}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[!] Failed to open camera.")
        print("    Check: ls /dev/video0")
        print("    Check: sudo systemctl restart nvargus-daemon")
        return

    # Display window
    view_w, view_h = 960, 540
    window_name = "X Detector + Motor Control"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, view_w, view_h)

    # State
    motors_running = False
    motor_start_time = 0
    motor_duration = args.duration
    fps_count = 0
    fps_start = time.time()
    display_fps = 0

    print(f"[*] Running (conf={args.conf}, throttle={args.throttle}%, "
          f"duration={args.duration}s)")
    print(f"[*] Press 'q' to quit, 's' to snapshot")
    print("-" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # --- YOLO inference on full 1920x1080 ---
            results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

            # --- Downscale for display ---
            display_frame = cv2.resize(frame, (view_w, view_h))
            scale_x = view_w / frame.shape[1]
            scale_y = view_h / frame.shape[0]
            display_frame, detections = draw_detections(
                display_frame, results, args.conf, scale_x, scale_y
            )

            det_count = len(detections)

            # --- Motor control logic ---
            if det_count > 0 and not motors_running:
                # X DETECTED — spin all motors!
                print(f"\n[!!!] X DETECTED! Starting motors...")
                for d in detections:
                    print(f"    {d['class']} conf={d['conf']:.2f} "
                          f"center={d['center']}")

                if not args.dry_run:
                    arm_and_spin_motors(master, args.throttle, args.duration)
                else:
                    print(f"[DRY RUN] Would spin all motors at "
                          f"{args.throttle}% for {args.duration}s")

                motors_running = True
                motor_start_time = time.time()

            # Check if motor duration expired
            if motors_running:
                elapsed_motor = time.time() - motor_start_time
                remaining = max(0, motor_duration - elapsed_motor)

                if elapsed_motor >= motor_duration:
                    print(f"\n[---] {args.duration}s elapsed — stopping motors")
                    if not args.dry_run:
                        force_disarm(master)
                    else:
                        print("[DRY RUN] Would disarm now")
                    motors_running = False

            # --- FPS ---
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                display_fps = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()

            # --- Status overlay on display ---
            if motors_running:
                remaining = max(0, motor_duration - (time.time() - motor_start_time))
                status = f"MOTORS ON {args.throttle}% | {remaining:.1f}s left"
                status_color = (0, 0, 255)  # red
                # Red border when motors active
                cv2.rectangle(display_frame, (0, 0),
                              (view_w - 1, view_h - 1), (0, 0, 255), 4)
            else:
                status = "STANDBY — waiting for X"
                status_color = (200, 200, 200)

            info = f"FPS:{display_fps:.1f} | X:{det_count} | {status}"
            cv2.putText(display_frame, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

            # Show detection centers
            if det_count > 0:
                for d in detections:
                    cx, cy = d['center']
                    dcx = int(cx * scale_x)
                    dcy = int(cy * scale_y)
                    cv2.circle(display_frame, (dcx, dcy), 8, (0, 0, 255), -1)
                    cv2.putText(display_frame,
                                f"({cx},{cy})", (dcx + 10, dcy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                fname = f"x_detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"[+] Saved: {fname}")

    except KeyboardInterrupt:
        print("\n[*] Interrupted.")

    finally:
        # Safety: always disarm on exit
        if motors_running and not args.dry_run:
            print("[*] Safety disarm on exit...")
            force_disarm(master)
        cap.release()
        cv2.destroyAllWindows()
        print("[*] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="X Target Detector + Motor Control (Pixhawk 6C)")

    # Detection
    parser.add_argument("--weights", default="best_22.pt",
                        help="YOLO weights (default: best_22.pt)")
    parser.add_argument("--mode", choices=["4k", "1080p"], default="1080p",
                        help="Camera mode (default: 1080p)")
    parser.add_argument("--conf", type=float, default=0.50,
                        help="Confidence threshold (default: 0.50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO inference size (default: 640)")

    # Pixhawk
    parser.add_argument("--device", default="/dev/ttyACM0",
                        help="Serial device (default: /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200,
                        help="Baud rate (default: 115200)")
    parser.add_argument("--throttle", type=int, default=100,
                        help="Motor throttle percent 0-100 (default: 100)")
    parser.add_argument("--duration", type=int, default=10,
                        help="Seconds to spin motors (default: 10)")

    # Safety
    parser.add_argument("--dry-run", action="store_true",
                        help="Camera + detection only, no motor control")

    args = parser.parse_args()

    print("=" * 60)
    print("  X DETECTOR + MOTOR CONTROL")
    print(f"  Throttle: {args.throttle}% | Duration: {args.duration}s")
    if args.dry_run:
        print("  *** DRY RUN — no motors ***")
    else:
        print(f"  Pixhawk: {args.device} @ {args.baud}")
    print("=" * 60)

    if args.throttle == 100 and not args.dry_run:
        print("\n[!] WARNING: 100% throttle! Props removed? (y/n)")
        if input().strip().lower() != 'y':
            print("[*] Aborted.")
            exit(0)

    run(args)
