#!/usr/bin/env python3
"""
X Target Detector + Centering Guide — Jetson Orin Nano + IMX477
================================================================
Shows live video on the connected monitor with:
  - Bounding box around detected X
  - Red/green dot at X center
  - Crosshair at frame center
  - Line connecting the two
  - Direction arrows on screen

Terminal also prints movement guidance.

Center-finding methods:
  1. BBOX center (default): YOLO bounding box midpoint.
  2. Refined (--refine): Green segmentation within bbox for precise crossing point.

Usage (run on Jetson with monitor):
    python3 x_detect_guide.py
    python3 x_detect_guide.py --refine
    python3 x_detect_guide.py --deadzone 80
    python3 x_detect_guide.py --headless          # terminal only, no window
    python3 x_detect_guide.py --snapshot           # single frame + save

Transfer:
    scp x_detect_guide.py best_22.pt jetson@<ip>:~/
"""

import argparse
import time
import cv2
import os
import numpy as np
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
# Load model
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
# Find the center of the X
# ---------------------------------------------------------------------------
def find_x_center_bbox(box):
    """Simple bounding box midpoint."""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    return (x1 + x2) // 2, (y1 + y2) // 2


def find_x_center_refined(frame, box):
    """
    Green-segmentation within YOLO bbox.
    Crops bbox, thresholds for green tape in HSV, computes centroid
    of green pixels (naturally lands at the crossing point).
    Falls back to bbox center if not enough green found.
    """
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

    # coords are (row, col) = (y, x)
    return x1c + int(np.mean(coords[:, 1])), y1c + int(np.mean(coords[:, 0]))


# ---------------------------------------------------------------------------
# Centering guidance
# ---------------------------------------------------------------------------
def compute_guidance(cx, cy, frame_w, frame_h, deadzone=50):
    mid_x = frame_w // 2
    mid_y = frame_h // 2
    dx = cx - mid_x   # positive = X is RIGHT of center
    dy = cy - mid_y   # positive = X is BELOW center

    if abs(dx) <= deadzone and abs(dy) <= deadzone:
        return "CENTERED", dx, dy, True

    parts = []
    if dx < -deadzone:
        parts.append(f"<< LEFT ({abs(dx)}px)")
    elif dx > deadzone:
        parts.append(f">> RIGHT ({abs(dx)}px)")
    else:
        parts.append("H:OK")

    if dy < -deadzone:
        parts.append(f"^^ UP ({abs(dy)}px)")
    elif dy > deadzone:
        parts.append(f"vv DOWN ({abs(dy)}px)")
    else:
        parts.append("V:OK")

    return " | ".join(parts), dx, dy, False


# ---------------------------------------------------------------------------
# Live mode — cv2.imshow on local monitor + terminal guidance
# ---------------------------------------------------------------------------
def run_live(args):
    print("[*] Loading X detection model...")
    model = load_model(args.weights)

    mode = MODES[args.mode]
    pipeline = build_gstreamer_pipeline(**mode)
    infer_w = mode["display_width"]
    infer_h = mode["display_height"]
    print(f"[*] Opening camera: {args.mode} -> {infer_w}x{infer_h}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[!] Failed to open camera.")
        print("    Check: ls /dev/video0")
        print("    Check: sudo systemctl restart nvargus-daemon")
        return

    view_w, view_h = 960, 540
    window_name = "X Detector — Centering Guide"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, view_w, view_h)

    fps_count = 0
    fps_start = time.time()
    display_fps = 0
    debug_saved = False

    method_tag = "REFINED" if args.refine else "BBOX"
    print(f"[*] Running (conf={args.conf}, imgsz={args.imgsz}, "
          f"method={method_tag}, deadzone={args.deadzone}px)")
    print(f"    Frame center: ({infer_w // 2}, {infer_h // 2})")
    print(f"    Press 'q' to quit, 's' to snapshot.")
    print("-" * 70)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            if args.save_debug and not debug_saved:
                cv2.imwrite("debug_raw_frame.jpg", frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"[DEBUG] Saved: debug_raw_frame.jpg "
                      f"({frame.shape[1]}x{frame.shape[0]})")
                debug_saved = True

            frame_h, frame_w = frame.shape[:2]

            # --- INFERENCE ON FULL-RES FRAME ---
            results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

            # downscale for display
            display_frame = cv2.resize(frame, (view_w, view_h))
            scale_x = view_w / frame_w
            scale_y = view_h / frame_h

            # --- DRAW FRAME CENTER CROSSHAIR ---
            smid_x = view_w // 2
            smid_y = view_h // 2
            cv2.line(display_frame, (smid_x - 25, smid_y), (smid_x + 25, smid_y),
                     (200, 200, 200), 1)
            cv2.line(display_frame, (smid_x, smid_y - 25), (smid_x, smid_y + 25),
                     (200, 200, 200), 1)
            cv2.circle(display_frame, (smid_x, smid_y), 6, (200, 200, 200), 1)

            # --- PROCESS DETECTIONS ---
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

                    # draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_name = result.names[int(box.cls[0])]
                    dx1 = int(x1 * scale_x)
                    dy1 = int(y1 * scale_y)
                    dx2 = int(x2 * scale_x)
                    dy2 = int(y2 * scale_y)

                    cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2),
                                  (0, 255, 0), 2)
                    label = f"{cls_name} {conf:.2f}"
                    lsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(display_frame, (dx1, dy1 - lsz[1] - 10),
                                  (dx1 + lsz[0], dy1), (0, 255, 0), -1)
                    cv2.putText(display_frame, label, (dx1, dy1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # --- GUIDANCE FOR BEST DETECTION ---
            if best_box is not None:
                # find X center
                if args.refine:
                    cx, cy = find_x_center_refined(frame, best_box)
                else:
                    cx, cy = find_x_center_bbox(best_box)

                direction_str, dx, dy, centered = compute_guidance(
                    cx, cy, frame_w, frame_h, args.deadzone
                )

                # draw X center dot
                dcx = int(cx * scale_x)
                dcy = int(cy * scale_y)
                color = (0, 255, 0) if centered else (0, 0, 255)
                cv2.circle(display_frame, (dcx, dcy), 8, color, -1)
                cv2.circle(display_frame, (dcx, dcy), 10, (255, 255, 255), 2)

                # line from X center to frame center
                cv2.line(display_frame, (dcx, dcy), (smid_x, smid_y), color, 2)

                # guidance text on screen
                if centered:
                    guide_text = "*** CENTERED ***"
                    guide_color = (0, 255, 0)
                else:
                    guide_text = direction_str
                    guide_color = (0, 165, 255)

                cv2.putText(display_frame, guide_text,
                            (10, view_h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, guide_color, 2)

                # offset readout
                offset_text = f"offset: ({dx:+d}, {dy:+d})px"
                cv2.putText(display_frame, offset_text,
                            (10, view_h - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # terminal output
                cls_name = best_result.names[int(best_box.cls[0])]
                if centered:
                    print(f"\r[** CENTERED **] {cls_name} conf={best_conf:.2f} "
                          f"X@({cx},{cy}) offset=({dx:+d},{dy:+d}) "
                          f"FPS={display_fps:.1f}       ", end="", flush=True)
                else:
                    print(f"\r[MOVE] {direction_str} | "
                          f"{cls_name} conf={best_conf:.2f} "
                          f"X@({cx},{cy}) ({dx:+d},{dy:+d}) "
                          f"FPS={display_fps:.1f}       ", end="", flush=True)
            else:
                print(f"\rSearching... FPS: {display_fps:.1f} | No X   ",
                      end="", flush=True)

            # --- FPS ---
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                display_fps = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()

            info = f"FPS: {display_fps:.1f} | X: {det_count} | {method_tag}"
            cv2.putText(display_frame, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                fname = f"x_guide_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"\n[+] Saved full-res: {fname}")

    finally:
        print()
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Headless mode (terminal only)
# ---------------------------------------------------------------------------
def run_headless(args):
    print("[*] Loading X detection model...")
    model = load_model(args.weights)

    mode = MODES[args.mode]
    pipeline = build_gstreamer_pipeline(**mode)
    infer_w = mode["display_width"]
    infer_h = mode["display_height"]
    print(f"[*] Opening camera: {args.mode} -> {infer_w}x{infer_h}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[!] Failed to open camera.")
        return

    fps_count = 0
    fps_start = time.time()
    display_fps = 0

    method_tag = "REFINED" if args.refine else "BBOX"
    print(f"[*] Headless (conf={args.conf}, imgsz={args.imgsz}, "
          f"method={method_tag}, deadzone={args.deadzone}px)")
    print(f"    Frame center: ({infer_w // 2}, {infer_h // 2})")
    print("-" * 70)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame_h, frame_w = frame.shape[:2]
            results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

            best_box = None
            best_conf = 0.0
            best_result = None

            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf >= args.conf and conf > best_conf:
                        best_conf = conf
                        best_box = box
                        best_result = result

            if best_box is not None:
                if args.refine:
                    cx, cy = find_x_center_refined(frame, best_box)
                else:
                    cx, cy = find_x_center_bbox(best_box)

                direction_str, dx, dy, centered = compute_guidance(
                    cx, cy, frame_w, frame_h, args.deadzone
                )
                cls_name = best_result.names[int(best_box.cls[0])]

                if centered:
                    print(f"\r[** CENTERED **] {cls_name} conf={best_conf:.2f} "
                          f"X@({cx},{cy}) offset=({dx:+d},{dy:+d}) "
                          f"FPS={display_fps:.1f}       ", end="", flush=True)
                else:
                    print(f"\r[MOVE] {direction_str} | "
                          f"{cls_name} conf={best_conf:.2f} "
                          f"X@({cx},{cy}) ({dx:+d},{dy:+d}) "
                          f"FPS={display_fps:.1f}       ", end="", flush=True)
            else:
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
    print(f"[*] Opening camera: {args.mode} -> "
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

    frame_h, frame_w = frame.shape[:2]
    results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

    annotated = frame.copy()
    mid_x, mid_y = frame_w // 2, frame_h // 2
    cv2.line(annotated, (mid_x - 30, mid_y), (mid_x + 30, mid_y), (200, 200, 200), 2)
    cv2.line(annotated, (mid_x, mid_y - 30), (mid_x, mid_y + 30), (200, 200, 200), 2)

    det_count = 0
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < args.conf:
                continue
            det_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = result.names[int(box.cls[0])]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

            if args.refine:
                cx, cy = find_x_center_refined(frame, box)
            else:
                cx, cy = find_x_center_bbox(box)

            direction_str, dx, dy, centered = compute_guidance(
                cx, cy, frame_w, frame_h, args.deadzone
            )

            color = (0, 255, 0) if centered else (0, 0, 255)
            cv2.circle(annotated, (cx, cy), 12, color, -1)
            cv2.circle(annotated, (cx, cy), 14, (255, 255, 255), 3)
            cv2.line(annotated, (cx, cy), (mid_x, mid_y), color, 2)

            label = f"{cls_name} {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            method_tag = "refined" if args.refine else "bbox"
            print(f"[+] {cls_name} conf={conf:.2f}")
            print(f"    bbox=[{x1},{y1},{x2},{y2}]")
            print(f"    X center ({method_tag}): ({cx}, {cy})")
            print(f"    Frame center: ({mid_x}, {mid_y})")
            print(f"    Offset: dx={dx:+d}  dy={dy:+d}")
            print(f"    >>> {direction_str}")

    fname = f"x_guide_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(fname, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\n[+] Detected {det_count} X target(s)")
    print(f"[+] Saved: {fname} ({frame_w}x{frame_h})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="X Target Detector + Centering Guide — Jetson")
    parser.add_argument("--weights", default="best_22.pt",
                        help="Path to YOLO weights (default: best_22.pt)")
    parser.add_argument("--mode", choices=["4k", "1080p"], default="1080p",
                        help="Camera mode (default: 1080p)")
    parser.add_argument("--conf", type=float, default=0.50,
                        help="Confidence threshold (default: 0.50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO inference size (default: 640)")
    parser.add_argument("--save-debug", action="store_true",
                        help="Save first raw frame for debugging")
    parser.add_argument("--refine", action="store_true",
                        help="Use green-segmentation within bbox for precise "
                             "X crossing point (default: bbox midpoint)")
    parser.add_argument("--deadzone", type=int, default=50,
                        help="Pixel deadzone for centered (default: 50)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--headless", action="store_true",
                       help="Terminal only, no video window")
    group.add_argument("--snapshot", action="store_true",
                       help="Single frame detection + save")

    args = parser.parse_args()

    if args.headless:
        run_headless(args)
    elif args.snapshot:
        run_snapshot(args)
    else:
        run_live(args)
