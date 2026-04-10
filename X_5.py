#!/usr/bin/env python3
"""
X Target Detector — Jetson Orin Nano + IMX477 (FIXED)
======================================================
Based on face_detect_test.py (which works perfectly) with these changes:
  1. Uses your custom best.pt instead of yolov8n.pt
  2. Feeds FULL 1920x1080 to model (matching your training images)
  3. Downscales to 960x540 for display AFTER inference
  4. imgsz=640 (YOLO default, better than 512)

The key insight: face_detect_test.py works at 960x540 because yolov8n
was trained on millions of images at every resolution. YOUR custom model
was trained on 1920x1080 snapshots, so it needs 1920x1080 input.

Usage:
    python3 x_detect_fixed.py --headless
    python3 x_detect_fixed.py --headless --conf 0.5
    python3 x_detect_fixed.py --snapshot
    python3 x_detect_fixed.py
    python3 x_detect_fixed.py --save-debug    # saves raw frame for comparison

Transfer:
    scp x_detect_fixed.py best.pt jetson@<ip>:~/
"""

import argparse
import time
import cv2
import os
from datetime import datetime


# ---------------------------------------------------------------------------
# GStreamer pipeline
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
    """
    Same structure as camera_view.py and face_detect_test.py.
    BUT: display_width/height set to FULL 1920x1080 so the model
    gets the same resolution as your training snapshots.
    """
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


# ---------------------------------------------------------------------------
# Camera modes
#   OLD (broken): display_width=960, display_height=540  ← HALF resolution!
#   NEW (fixed):  display_width=1920, display_height=1080 ← matches training
# ---------------------------------------------------------------------------
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
# Load X detection model
# ---------------------------------------------------------------------------
def load_model(weights="best_22.pt"):
    from ultralytics import YOLO

    if not os.path.exists(weights):
        print(f"[!] Model not found: {weights}")
        print(f"    scp best.pt jetson@<ip>:~/")
        exit(1)

    model = YOLO(weights)
    print(f"[+] : {weights}")
    print(f"    Classes: {model.names}")
    return model


# ---------------------------------------------------------------------------
# Draw detections — same style as face_detect_test.py
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

            # remap coords for display frame
            dx1 = int(x1 * scale_x)
            dy1 = int(y1 * scale_y)
            dx2 = int(x2 * scale_x)
            dy2 = int(y2 * scale_y)

            # green box + label (same style as face_detect_test.py)
            cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (dx1, dy1 - label_size[1] - 10),
                          (dx1 + label_size[0], dy1), (0, 255, 0), -1)
            cv2.putText(frame, label, (dx1, dy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame, detections


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------
def run_live(args):
    print("[*] Loading X detection model...")
    model = load_model(args.weights)

    mode = MODES[args.mode]
    pipeline = build_gstreamer_pipeline(**mode)
    print(f"[*] Opening camera: {args.mode} → "
          f"{mode['display_width']}x{mode['display_height']} (full-res to model)")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[!] Failed to open camera.")
        print("    Check: ls /dev/video0")
        print("    Check: sudo systemctl restart nvargus-daemon")
        return

    view_w, view_h = 960, 540
    window_name = "X Detector — Fixed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, view_w, view_h)

    fps_count = 0
    fps_start = time.time()
    display_fps = 0
    debug_saved = False

    print(f"[*] Running (conf={args.conf}, imgsz={args.imgsz}). Press 'q' to quit, 's' to snapshot.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # save one debug frame
            if args.save_debug and not debug_saved:
                cv2.imwrite("debug_raw_frame.jpg", frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"[DEBUG] Saved: debug_raw_frame.jpg "
                      f"({frame.shape[1]}x{frame.shape[0]})")
                debug_saved = True

            # --- INFERENCE ON FULL 1920x1080 FRAME ---
            # Same call style as face_detect_test.py but with imgsz and conf
            results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

            # Downscale for display AFTER inference
            display_frame = cv2.resize(frame, (view_w, view_h))
            scale_x = view_w / frame.shape[1]
            scale_y = view_h / frame.shape[0]
            display_frame, det_count = draw_detections(
                display_frame, results, args.conf, scale_x, scale_y
            )

            # FPS
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                display_fps = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()

            info = f"FPS: {display_fps:.1f} | X: {det_count} | {args.mode} full→model"
            cv2.putText(display_frame, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                fname = f"x_detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"[+] Saved full-res: {fname}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Headless mode
# ---------------------------------------------------------------------------
def run_headless(args):
    print("[*] Loading X detection model...")
    model = load_model(args.weights)

    mode = MODES[args.mode]
    pipeline = build_gstreamer_pipeline(**mode)
    print(f"[*] Opening camera: {args.mode} → "
          f"{mode['display_width']}x{mode['display_height']} (full-res to model)")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[!] Failed to open camera.")
        print("    Check: ls /dev/video0")
        print("    Check: sudo systemctl restart nvargus-daemon")
        return

    fps_count = 0
    fps_start = time.time()
    display_fps = 0
    debug_saved = False

    print(f"[*] Headless (conf={args.conf}, imgsz={args.imgsz}). Ctrl+C to stop.")
    print("-" * 60)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # save one debug frame
            if args.save_debug and not debug_saved:
                cv2.imwrite("debug_raw_frame.jpg", frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"[DEBUG] Saved: debug_raw_frame.jpg "
                      f"({frame.shape[1]}x{frame.shape[0]}, ch={frame.shape[2]})")
                debug_saved = True

            # --- INFERENCE ON FULL 1920x1080 FRAME ---
            results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

            det_count = 0
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf >= args.conf:
                        det_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        cls_name = result.names[int(box.cls[0])]
                        print(f"\r[X FOUND] {cls_name} conf={conf:.2f} "
                              f"center=({cx},{cy}) "
                              f"bbox=[{x1},{y1},{x2},{y2}] "
                              f"FPS={display_fps:.1f}    ")

            if det_count == 0:
                print(f"\rSearching... FPS: {display_fps:.1f} | No X   ",
                      end="", flush=True)

            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                display_fps = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()

    except KeyboardInterrupt:
        print("\n[*] Stopped.")
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Snapshot mode
# ---------------------------------------------------------------------------
def run_snapshot(args):
    print("[*] Loading X detection model...")
    model = load_model(args.weights)

    mode = MODES[args.mode]
    pipeline = build_gstreamer_pipeline(**mode)
    print(f"[*] Opening camera: {args.mode} → "
          f"{mode['display_width']}x{mode['display_height']}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[!] Failed to open camera.")
        return

    print("[*] Warming up camera...")
    for _ in range(30):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("[!] Failed to capture frame.")
        return

    results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
    annotated = frame.copy()
    annotated, det_count = draw_detections(annotated, results, args.conf)

    fname = f"x_detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(fname, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[+] Detected {det_count} X target(s)")
    print(f"[+] Saved: {fname} ({frame.shape[1]}x{frame.shape[0]})")

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf >= args.conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cls_name = result.names[int(box.cls[0])]
                print(f"    {cls_name} ({conf:.2f}) "
                      f"center=({cx},{cy}) bbox=[{x1},{y1},{x2},{y2}]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="X Target Detector — Jetson (Fixed: full-res inference)")
    parser.add_argument("--weights", default="best_22.pt",
                        help="Path to YOLO weights (default: best.pt)")
    parser.add_argument("--mode", choices=["4k", "1080p"], default="1080p",
                        help="Camera mode (default: 1080p)")
    parser.add_argument("--conf", type=float, default=0.50,
                        help="Confidence threshold (default: 0.50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO inference size (default: 640)")
    parser.add_argument("--save-debug", action="store_true",
                        help="Save first raw frame for debugging")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--headless", action="store_true",
                       help="No display, terminal output (SSH)")
    group.add_argument("--snapshot", action="store_true",
                       help="Single frame detection + save")

    args = parser.parse_args()

    if args.headless:
        run_headless(args)
    elif args.snapshot:
        run_snapshot(args)
    else:
        run_live(args)
