#!/usr/bin/env python3
"""
X Target Detector + Centering Guide — Jetson Orin Nano + IMX477
================================================================
Modes:
  --stream       Video in browser at http://<jetson-ip>:5000, guidance in terminal
  --headless     Terminal only, no video
  --snapshot     Single frame, saves annotated image

The stream mode uses Flask MJPEG — no GTK/display needed. Just open
the URL on any device on the same network.

Center-finding methods:
  1. BBOX center (default): Uses YOLO bounding box midpoint.
  2. Refined (--refine): Green segmentation within bbox for precise crossing point.

Usage:
    python3 x_detect_guide.py --stream
    python3 x_detect_guide.py --stream --refine
    python3 x_detect_guide.py --stream --refine --deadzone 80
    python3 x_detect_guide.py --headless
    python3 x_detect_guide.py --snapshot
    python3 x_detect_guide.py --save-debug

Install (one time):
    pip3 install flask

Transfer:
    scp x_detect_guide.py best_22.pt jetson@<ip>:~/
"""

import argparse
import time
import cv2
import os
import numpy as np
import threading
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
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cx, cy


def find_x_center_refined(frame, box):
    """
    Green-segmentation within YOLO bbox.
    Crops bbox, thresholds for green tape in HSV, computes centroid.
    Falls back to bbox center if not enough green pixels found.
    """
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    h, w = frame.shape[:2]
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(w, x2)
    y2c = min(h, y2)

    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return find_x_center_bbox(box)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    coords = np.column_stack(np.where(mask > 0))
    if len(coords) < 50:
        return find_x_center_bbox(box)

    centroid_y = int(np.mean(coords[:, 0]))
    centroid_x = int(np.mean(coords[:, 1]))

    cx = x1c + centroid_x
    cy = y1c + centroid_y
    return cx, cy


# ---------------------------------------------------------------------------
# Centering guidance
# ---------------------------------------------------------------------------
def compute_guidance(cx, cy, frame_w, frame_h, deadzone=50):
    """
    Returns:
        direction_str, dx, dy, centered
    """
    mid_x = frame_w // 2
    mid_y = frame_h // 2

    dx = cx - mid_x  # positive = X is RIGHT of center
    dy = cy - mid_y  # positive = X is BELOW center

    if abs(dx) <= deadzone and abs(dy) <= deadzone:
        return "CENTERED", dx, dy, True

    parts = []

    if dx < -deadzone:
        parts.append(f"<< LEFT  ({abs(dx)}px)")
    elif dx > deadzone:
        parts.append(f">> RIGHT ({abs(dx)}px)")
    else:
        parts.append("H:OK")

    if dy < -deadzone:
        parts.append(f"^^ UP    ({abs(dy)}px)")
    elif dy > deadzone:
        parts.append(f"vv DOWN  ({abs(dy)}px)")
    else:
        parts.append("V:OK")

    return "  |  ".join(parts), dx, dy, False


# ---------------------------------------------------------------------------
# Annotate a frame for display (bboxes, center dot, crosshair, guide line)
# ---------------------------------------------------------------------------
def annotate_frame(display_frame, results, frame_w, frame_h,
                   conf_thresh=0.50, scale_x=1.0, scale_y=1.0,
                   refine=False, raw_frame=None, deadzone=50,
                   display_fps=0.0):
    """
    Draws on display_frame:
      - frame center crosshair
      - all bounding boxes
      - X center dot (green=centered, red=off)
      - line from X center to frame center
      - FPS overlay

    Returns: (annotated_frame, det_count, guidance_info)
        guidance_info = (cx, cy, dx, dy, direction_str, centered) or None
    """
    guidance_info = None

    # frame center crosshair
    smid_x = int((frame_w / 2) * scale_x)
    smid_y = int((frame_h / 2) * scale_y)
    cv2.line(display_frame, (smid_x - 25, smid_y), (smid_x + 25, smid_y),
             (200, 200, 200), 1)
    cv2.line(display_frame, (smid_x, smid_y - 25), (smid_x, smid_y + 25),
             (200, 200, 200), 1)
    cv2.circle(display_frame, (smid_x, smid_y), 6, (200, 200, 200), 1)

    # find best detection
    best_box = None
    best_conf = 0.0
    best_result = None
    detections = 0

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue
            detections += 1
            if conf > best_conf:
                best_conf = conf
                best_box = box
                best_result = result

            # draw bbox
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]

            dx1 = int(x1 * scale_x)
            dy1 = int(y1 * scale_y)
            dx2 = int(x2 * scale_x)
            dy2 = int(y2 * scale_y)

            cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (dx1, dy1 - label_size[1] - 10),
                          (dx1 + label_size[0], dy1), (0, 255, 0), -1)
            cv2.putText(display_frame, label, (dx1, dy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # guidance for best detection
    if best_box is not None:
        if refine and raw_frame is not None:
            cx, cy = find_x_center_refined(raw_frame, best_box)
        else:
            cx, cy = find_x_center_bbox(best_box)

        direction_str, dx, dy, centered = compute_guidance(
            cx, cy, frame_w, frame_h, deadzone
        )
        guidance_info = (cx, cy, dx, dy, direction_str, centered)

        # draw X center dot on display
        dcx = int(cx * scale_x)
        dcy = int(cy * scale_y)
        color = (0, 255, 0) if centered else (0, 0, 255)
        cv2.circle(display_frame, (dcx, dcy), 8, color, -1)
        cv2.circle(display_frame, (dcx, dcy), 10, (255, 255, 255), 2)

        # line from X center to frame center
        cv2.line(display_frame, (dcx, dcy), (smid_x, smid_y), color, 2)

    # FPS overlay
    method_tag = "refined" if refine else "bbox"
    info = f"FPS: {display_fps:.1f} | X: {detections} | {method_tag}"
    cv2.putText(display_frame, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return display_frame, detections, guidance_info


# ---------------------------------------------------------------------------
# Stream mode — Flask MJPEG server + terminal guidance
# ---------------------------------------------------------------------------
def run_stream(args):
    from flask import Flask, Response

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

    # shared state between capture thread and flask
    lock = threading.Lock()
    latest_jpeg = [None]
    running = [True]

    view_w, view_h = 960, 540

    fps_count = [0]
    fps_start = [time.time()]
    display_fps = [0.0]
    debug_saved = [False]

    method_tag = "REFINED" if args.refine else "BBOX"
    print(f"[*] Stream mode (conf={args.conf}, imgsz={args.imgsz}, "
          f"method={method_tag}, deadzone={args.deadzone}px)")
    print(f"    Frame center: ({infer_w // 2}, {infer_h // 2})")
    print(f"")
    print(f"    ============================================")
    print(f"    Open in browser: http://<jetson-ip>:{args.port}")
    print(f"    ============================================")
    print(f"")
    print("-" * 70)

    def capture_loop():
        while running[0]:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            if args.save_debug and not debug_saved[0]:
                cv2.imwrite("debug_raw_frame.jpg", frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"[DEBUG] Saved: debug_raw_frame.jpg "
                      f"({frame.shape[1]}x{frame.shape[0]})")
                debug_saved[0] = True

            frame_h_px, frame_w_px = frame.shape[:2]

            # inference on full-res
            results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

            # build annotated display frame
            display_frame = cv2.resize(frame, (view_w, view_h))
            scale_x = view_w / frame_w_px
            scale_y = view_h / frame_h_px

            display_frame, det_count, guidance = annotate_frame(
                display_frame, results,
                frame_w=frame_w_px, frame_h=frame_h_px,
                conf_thresh=args.conf,
                scale_x=scale_x, scale_y=scale_y,
                refine=args.refine, raw_frame=frame,
                deadzone=args.deadzone,
                display_fps=display_fps[0]
            )

            # encode to JPEG for streaming
            _, buf = cv2.imencode('.jpg', display_frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, 80])
            with lock:
                latest_jpeg[0] = buf.tobytes()

            # terminal guidance
            if guidance is not None:
                cx, cy, dx, dy, direction_str, centered = guidance
                if centered:
                    print(f"\r[** CENTERED **] "
                          f"X@({cx},{cy}) offset=({dx:+d},{dy:+d}) "
                          f"FPS={display_fps[0]:.1f}       ", end="", flush=True)
                else:
                    print(f"\r[MOVE] {direction_str}  "
                          f"X@({cx},{cy}) ({dx:+d},{dy:+d}) "
                          f"FPS={display_fps[0]:.1f}       ", end="", flush=True)
            else:
                print(f"\rSearching... FPS: {display_fps[0]:.1f} "
                      f"| No X detected          ", end="", flush=True)

            # FPS
            fps_count[0] += 1
            elapsed = time.time() - fps_start[0]
            if elapsed >= 1.0:
                display_fps[0] = fps_count[0] / elapsed
                fps_count[0] = 0
                fps_start[0] = time.time()

        cap.release()

    # start capture in background thread
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()

    # Flask app
    app = Flask(__name__)

    @app.route('/')
    def index():
        return """
        <html>
        <head>
            <title>X Detector - Live</title>
            <style>
                body {
                    background: #111; margin: 0; padding: 0;
                    display: flex; justify-content: center; align-items: center;
                    min-height: 100vh; font-family: monospace;
                }
                img {
                    max-width: 100%; height: auto;
                    border: 2px solid #333;
                }
            </style>
        </head>
        <body>
            <img src="/video_feed" />
        </body>
        </html>
        """

    def generate():
        while running[0]:
            with lock:
                frame_bytes = latest_jpeg[0]
            if frame_bytes is None:
                time.sleep(0.05)
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + frame_bytes + b'\r\n')
            time.sleep(0.03)  # ~30 fps max to browser

    @app.route('/video_feed')
    def video_feed():
        return Response(generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    try:
        # suppress Flask request logs to keep terminal clean for guidance
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        app.run(host='0.0.0.0', port=args.port, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        running[0] = False
        print("\n[*] Stopped.")


# ---------------------------------------------------------------------------
# Headless mode (terminal only, no video)
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
    debug_saved = False

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

            if args.save_debug and not debug_saved:
                cv2.imwrite("debug_raw_frame.jpg", frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"[DEBUG] Saved: debug_raw_frame.jpg "
                      f"({frame.shape[1]}x{frame.shape[0]})")
                debug_saved = True

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
                    print(f"\r[MOVE] {direction_str}  |  "
                          f"{cls_name} conf={best_conf:.2f} "
                          f"X@({cx},{cy}) ({dx:+d},{dy:+d}) "
                          f"FPS={display_fps:.1f}       ", end="", flush=True)
            else:
                print(f"\rSearching... FPS: {display_fps:.1f} | No X detected   ",
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
                        help="Pixel deadzone for 'centered' (default: 50)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Flask server port (default: 5000)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stream", action="store_true",
                       help="Video in browser + guidance in terminal")
    group.add_argument("--headless", action="store_true",
                       help="Terminal only, no video")
    group.add_argument("--snapshot", action="store_true",
                       help="Single frame detection + save")

    args = parser.parse_args()

    if args.stream:
        run_stream(args)
    elif args.headless:
        run_headless(args)
    elif args.snapshot:
        run_snapshot(args)
    else:
        # default to stream mode since live GTK won't work over SSH
        print("[*] No mode specified, defaulting to --stream")
        args.stream = True
        run_stream(args)
