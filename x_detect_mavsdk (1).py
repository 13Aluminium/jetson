#!/usr/bin/env python3
"""
X Target Detector + MAVSDK Motor Control
=========================================
Jetson Orin Nano + IMX477 + Pixhawk 6C

When an X target is detected for N consecutive frames, arms the drone
and starts motors via MAVSDK offboard control.

Hardware wiring (Pixhawk 6C → Jetson Orin Nano):
  Option A: USB cable  → connection: "serial:///dev/ttyACM0:57600"
  Option B: UART/TELEM → connection: "serial:///dev/ttyTHS1:57600"

Prerequisites:
    pip install mavsdk ultralytics opencv-python

Usage:
    python3 x_detect_mavsdk.py --headless --connection serial:///dev/ttyACM0:57600
    python3 x_detect_mavsdk.py --headless --connection serial:///dev/ttyTHS1:921600
    python3 x_detect_mavsdk.py --headless --connection udp://:14540  # SITL testing
    python3 x_detect_mavsdk.py  # GUI mode with default USB connection

Safety:
    - Requires X detected for --arm-frames consecutive frames (default: 10)
    - Disarms automatically when X is lost for --disarm-frames frames (default: 30)
    - Press 'q' or Ctrl+C to disarm and quit
    - --dry-run mode to test detection logic without arming
"""

import argparse
import asyncio
import time
import cv2
import os
from datetime import datetime

# MAVSDK imports
from mavsdk import System
from mavsdk.offboard import OffboardError, Attitude


# ---------------------------------------------------------------------------
# GStreamer pipeline (unchanged from your working script)
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
        exit(1)

    model = YOLO(weights)
    print(f"[+] Model loaded: {weights}")
    print(f"    Classes: {model.names}")
    return model


# ---------------------------------------------------------------------------
# Draw detections
# ---------------------------------------------------------------------------
def draw_detections(frame, results, conf_thresh=0.50, scale_x=1.0, scale_y=1.0):
    detections = 0
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue

            detections += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]

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
# MAVSDK: Connect to Pixhawk
# ---------------------------------------------------------------------------
async def connect_pixhawk(connection_string):
    """Connect to Pixhawk 6C and return the System object."""
    drone = System()
    print(f"[MAV] Connecting to Pixhawk at: {connection_string}")
    await drone.connect(system_address=connection_string)

    # Wait for connection
    print("[MAV] Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[MAV] ✓ Drone connected!")
            break

    # Wait for global position estimate (needed for arming)
    print("[MAV] Waiting for position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("[MAV] ✓ Position estimate OK")
            break

    return drone


# ---------------------------------------------------------------------------
# MAVSDK: Arm and start offboard with throttle
# ---------------------------------------------------------------------------
async def arm_and_start_motors(drone, throttle_pct=0.15):
    """
    Arm the drone and set a low throttle via offboard Attitude control.

    throttle_pct: 0.0 to 1.0 — start LOW (0.10-0.20) for safety!
                  This spins the motors but won't lift off at low values.
    """
    try:
        # Set initial setpoint BEFORE starting offboard (required by PX4)
        # Attitude: roll=0, pitch=0, yaw=0, thrust=throttle_pct
        await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, throttle_pct))

        # Arm the drone
        print(f"[MAV] >>> ARMING DRONE <<<")
        await drone.action.arm()
        print(f"[MAV] ✓ Armed")

        # Start offboard mode
        print(f"[MAV] Starting offboard mode (throttle={throttle_pct*100:.0f}%)")
        await drone.offboard.start()
        print(f"[MAV] ✓ Offboard active — motors spinning")

        return True

    except Exception as e:
        print(f"[MAV] ✗ Arm/offboard failed: {e}")
        return False


# ---------------------------------------------------------------------------
# MAVSDK: Disarm safely
# ---------------------------------------------------------------------------
async def disarm_drone(drone):
    """Stop offboard and disarm."""
    try:
        print("[MAV] Stopping offboard...")
        # Set throttle to 0 first
        await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.0))
        await asyncio.sleep(0.5)
        await drone.offboard.stop()
    except OffboardError as e:
        print(f"[MAV] Offboard stop note: {e}")
    except Exception:
        pass

    try:
        print("[MAV] Disarming...")
        await drone.action.disarm()
        print("[MAV] ✓ Disarmed")
    except Exception as e:
        print(f"[MAV] Disarm note: {e}")


# ---------------------------------------------------------------------------
# Main async loop: detection + motor control
# ---------------------------------------------------------------------------
async def run_detection_loop(args):
    # --- Load model ---
    print("[*] Loading X detection model...")
    model = load_model(args.weights)

    # --- Connect to Pixhawk ---
    drone = None
    if not args.dry_run:
        drone = await connect_pixhawk(args.connection)
    else:
        print("[DRY RUN] Skipping Pixhawk connection")

    # --- Open camera ---
    mode = MODES[args.mode]
    pipeline = build_gstreamer_pipeline(**mode)
    print(f"[*] Opening camera: {args.mode} → "
          f"{mode['display_width']}x{mode['display_height']}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[!] Failed to open camera.")
        return

    # --- State tracking ---
    consecutive_detections = 0     # frames with X detected
    consecutive_no_detections = 0  # frames without X
    motors_armed = False
    view_w, view_h = 960, 540

    # GUI setup (non-headless)
    if not args.headless:
        window_name = "X Detector + MAVSDK"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, view_w, view_h)

    fps_count = 0
    fps_start = time.time()
    display_fps = 0

    print(f"[*] Running (conf={args.conf}, arm_after={args.arm_frames} frames, "
          f"disarm_after={args.disarm_frames} frames)")
    print(f"[*] Throttle on arm: {args.throttle*100:.0f}%")
    print("-" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.05)
                continue

            # --- Run YOLO inference on full-res frame ---
            results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

            # Count detections
            det_count = 0
            for result in results:
                for box in result.boxes:
                    if float(box.conf[0]) >= args.conf:
                        det_count += 1

            # --- Detection state machine ---
            if det_count > 0:
                consecutive_detections += 1
                consecutive_no_detections = 0

                # ARM: X detected for enough consecutive frames
                if (consecutive_detections >= args.arm_frames
                        and not motors_armed):
                    print(f"\n[!!!] X CONFIRMED ({consecutive_detections} frames) "
                          f"— STARTING MOTORS")
                    if not args.dry_run:
                        success = await arm_and_start_motors(
                            drone, throttle_pct=args.throttle
                        )
                        motors_armed = success
                    else:
                        print("[DRY RUN] Would arm motors now")
                        motors_armed = True

            else:
                consecutive_no_detections += 1
                consecutive_detections = 0

                # DISARM: X lost for enough frames
                if (consecutive_no_detections >= args.disarm_frames
                        and motors_armed):
                    print(f"\n[---] X LOST ({consecutive_no_detections} frames) "
                          f"— STOPPING MOTORS")
                    if not args.dry_run:
                        await disarm_drone(drone)
                    else:
                        print("[DRY RUN] Would disarm motors now")
                    motors_armed = False

            # --- FPS ---
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                display_fps = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()

            # --- Status indicator ---
            armed_str = "ARMED ●" if motors_armed else "DISARMED ○"
            det_progress = (f"det:{consecutive_detections}/{args.arm_frames}"
                            if not motors_armed
                            else f"lost:{consecutive_no_detections}/{args.disarm_frames}")

            # --- Display ---
            if args.headless:
                if det_count > 0:
                    for result in results:
                        for box in result.boxes:
                            conf = float(box.conf[0])
                            if conf >= args.conf:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cx = (x1 + x2) // 2
                                cy = (y1 + y2) // 2
                                cls_name = result.names[int(box.cls[0])]
                                print(f"\r[X] {cls_name} {conf:.2f} "
                                      f"({cx},{cy}) | {armed_str} | "
                                      f"{det_progress} | FPS:{display_fps:.1f}   ")
                else:
                    print(f"\rSearching... {armed_str} | {det_progress} | "
                          f"FPS:{display_fps:.1f}   ", end="", flush=True)
            else:
                # GUI mode
                display_frame = cv2.resize(frame, (view_w, view_h))
                scale_x = view_w / frame.shape[1]
                scale_y = view_h / frame.shape[0]
                display_frame, _ = draw_detections(
                    display_frame, results, args.conf, scale_x, scale_y
                )

                # Status bar
                color = (0, 0, 255) if motors_armed else (200, 200, 200)
                info = (f"FPS:{display_fps:.1f} | X:{det_count} | "
                        f"{armed_str} | {det_progress}")
                cv2.putText(display_frame, info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord('s'):
                    fname = f"x_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    print(f"[+] Saved: {fname}")

            # Yield to async event loop
            await asyncio.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")

    finally:
        # --- ALWAYS disarm on exit ---
        if motors_armed and not args.dry_run:
            print("[*] Safety disarm on exit...")
            await disarm_drone(drone)
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print("[*] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="X Target Detector + MAVSDK Motor Control")

    # Detection args (same as before)
    parser.add_argument("--weights", default="best_22.pt",
                        help="YOLO weights (default: best_22.pt)")
    parser.add_argument("--mode", choices=["4k", "1080p"], default="1080p",
                        help="Camera mode (default: 1080p)")
    parser.add_argument("--conf", type=float, default=0.50,
                        help="Confidence threshold (default: 0.50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO inference size (default: 640)")

    # MAVSDK args
    parser.add_argument("--connection", default="serial:///dev/ttyACM0:57600",
                        help="MAVSDK connection string (default: serial USB)")
    parser.add_argument("--throttle", type=float, default=0.15,
                        help="Motor throttle 0.0-1.0 when armed (default: 0.15)")
    parser.add_argument("--arm-frames", type=int, default=10,
                        help="Consecutive detection frames before arming (default: 10)")
    parser.add_argument("--disarm-frames", type=int, default=30,
                        help="Consecutive no-detection frames before disarming (default: 30)")

    # Mode args
    parser.add_argument("--headless", action="store_true",
                        help="No display, terminal output only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test detection logic without connecting to Pixhawk")

    args = parser.parse_args()

    # Safety check
    if args.throttle > 0.5:
        print(f"[!] WARNING: throttle={args.throttle} is HIGH. "
              f"Are you sure? (y/n)")
        if input().strip().lower() != 'y':
            print("[*] Aborted.")
            exit(0)

    asyncio.run(run_detection_loop(args))
