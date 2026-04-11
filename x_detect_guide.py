#!/usr/bin/env python3
"""
X Target Detector + Centering Guide — Jetson Orin Nano + IMX477
================================================================
Extended from x_detect_fixed.py with centering guidance:
  - Detects the X target using your custom YOLO model
  - Computes the center of the X (two methods available)
  - Prints directional guidance to center the X on screen

Center-finding methods:
  1. BBOX center (default): Uses YOLO bounding box midpoint. Fast, reliable
     when the full X is in frame.
  2. Refined (--refine): Crops the YOLO bbox, does HSV green segmentation
     within it, computes the centroid of green pixels. Finds the actual
     crossing point of the tape. More precise, handles partial occlusion
     of one arm better.

Usage:
    python3 x_detect_guide.py --headless
    python3 x_detect_guide.py --headless --refine
    python3 x_detect_guide.py --headless --deadzone 80
    python3 x_detect_guide.py --headless --refine --conf 0.4
    python3 x_detect_guide.py --snapshot
    python3 x_detect_guide.py                     # live window mode
    python3 x_detect_guide.py --save-debug

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
# GStreamer pipeline (unchanged from original)
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
# Load model (unchanged)
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
# Find the center of the X — two methods
# ---------------------------------------------------------------------------
def find_x_center_bbox(box):
    """
    Method 1: Simple bounding box midpoint.
    Fast, works well when the full X is visible.
    """
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cx, cy


def find_x_center_refined(frame, box):
    """
    Method 2: Green-segmentation within YOLO bbox.

    Steps:
      1. Crop the frame to the YOLO bounding box (isolates the X region,
         excludes background green objects like shirts)
      2. Convert crop to HSV
      3. Threshold for green (the tape color)
      4. Compute the centroid of green pixels — this naturally falls at
         the crossing point where both arms overlap (highest density)
      5. Map back to full-frame coordinates

    Falls back to bbox center if segmentation finds nothing (e.g., weird
    lighting washes out the green).
    """
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # clamp to frame bounds
    h, w = frame.shape[:2]
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(w, x2)
    y2c = min(h, y2)

    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return find_x_center_bbox(box)

    # convert to HSV and threshold for green tape
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # green tape range — fairly broad to handle varying lighting
    # adjust these if your tape looks different under different lights
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # optional: clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # find centroid of green pixels
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) < 50:
        # not enough green pixels found — fall back to bbox center
        return find_x_center_bbox(box)

    # coords are (row, col) = (y, x)
    centroid_y = int(np.mean(coords[:, 0]))
    centroid_x = int(np.mean(coords[:, 1]))

    # map back to full-frame coordinates
    cx = x1c + centroid_x
    cy = y1c + centroid_y

    return cx, cy


# ---------------------------------------------------------------------------
# Centering guidance
# ---------------------------------------------------------------------------
def compute_guidance(cx, cy, frame_w, frame_h, deadzone=50):
    """
    Compare X center to frame center, return guidance string and offsets.

    Returns:
        direction_str:  e.g. "← LEFT  ↑ UP"  or  "✓ CENTERED"
        dx, dy:         signed pixel offsets (positive = X is right/below center)
        centered:       bool, True if within deadzone on both axes
    """
    mid_x = frame_w // 2
    mid_y = frame_h // 2

    dx = cx - mid_x   # positive = X is to the RIGHT of center
    dy = cy - mid_y   # positive = X is BELOW center

    parts = []

    if abs(dx) <= deadzone and abs(dy) <= deadzone:
        return "✓ CENTERED", dx, dy, True

    # horizontal
    if dx < -deadzone:
        parts.append(f"← LEFT  ({abs(dx)}px)")
    elif dx > deadzone:
        parts.append(f"→ RIGHT ({abs(dx)}px)")
    else:
        parts.append("↔ H:OK")

    # vertical
    if dy < -deadzone:
        parts.append(f"↑ UP    ({abs(dy)}px)")
    elif dy > deadzone:
        parts.append(f"↓ DOWN  ({abs(dy)}px)")
    else:
        parts.append("↕ V:OK")

    return "  |  ".join(parts), dx, dy, False


# ---------------------------------------------------------------------------
# Draw detections + crosshair (for live/snapshot modes)
# ---------------------------------------------------------------------------
def draw_detections_with_guide(
    frame, results, frame_w, frame_h, conf_thresh=0.50,
    scale_x=1.0, scale_y=1.0, refine=False, raw_frame=None,
    deadzone=50
):
    """
    Draws bounding boxes, the X center, frame crosshair, and guidance arrow.
    Returns the annotated frame, detection count, and guidance info.
    """
    detections = 0
    guidance_info = None

    # draw frame center crosshair (faint)
    smid_x = int((frame_w / 2) * scale_x)
    smid_y = int((frame_h / 2) * scale_y)
    cv2.line(frame, (smid_x - 20, smid_y), (smid_x + 20, smid_y), (128, 128, 128), 1)
    cv2.line(frame, (smid_x, smid_y - 20), (smid_x, smid_y + 20), (128, 128, 128), 1)

    # pick highest-confidence detection
    best_box = None
    best_conf = 0.0
    best_result = None

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

            # draw all bounding boxes
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

    # compute guidance for the best detection
    if best_box is not None:
        if refine and raw_frame is not None:
            cx, cy = find_x_center_refined(raw_frame, best_box)
        else:
            cx, cy = find_x_center_bbox(best_box)

        direction_str, dx, dy, centered = compute_guidance(
            cx, cy, frame_w, frame_h, deadzone
        )
        guidance_info = (cx, cy, dx, dy, direction_str, centered)

        # draw X center point on display frame
        dcx = int(cx * scale_x)
        dcy = int(cy * scale_y)
        color = (0, 255, 0) if centered else (0, 0, 255)
        cv2.circle(frame, (dcx, dcy), 8, color, -1)
        cv2.circle(frame, (dcx, dcy), 10, (255, 255, 255), 2)

        # draw line from X center to frame center
        cv2.line(frame, (dcx, dcy), (smid_x, smid_y), color, 2)

        # guidance text on frame
        guide_color = (0, 255, 0) if centered else (0, 165, 255)
        cv2.putText(frame, direction_str, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, guide_color, 2)

    return frame, detections, guidance_info


# ---------------------------------------------------------------------------
# Live mode (with guidance)
# ---------------------------------------------------------------------------
def run_live(args):
    print("[*] Loading X detection model...")
    model = load_model(args.weights)

    mode = MODES[args.mode]
    pipeline = build_gstreamer_pipeline(**mode)
    infer_w = mode["display_width"]
    infer_h = mode["display_height"]
    print(f"[*] Opening camera: {args.mode} → {infer_w}x{infer_h}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[!] Failed to open camera.")
        return

    view_w, view_h = 960, 540
    window_name = "X Detector — Centering Guide"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, view_w, view_h)

    fps_count = 0
    fps_start = time.time()
    display_fps = 0
    debug_saved = False

    print(f"[*] Running (conf={args.conf}, imgsz={args.imgsz}, "
          f"refine={args.refine}, deadzone={args.deadzone}px)")
    print(f"    Press 'q' to quit, 's' to snapshot.")
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

            # inference on full-res frame
            results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

            # downscale for display
            display_frame = cv2.resize(frame, (view_w, view_h))
            scale_x = view_w / frame.shape[1]
            scale_y = view_h / frame.shape[0]

            display_frame, det_count, guidance = draw_detections_with_guide(
                display_frame, results,
                frame_w=frame.shape[1], frame_h=frame.shape[0],
                conf_thresh=args.conf,
                scale_x=scale_x, scale_y=scale_y,
                refine=args.refine, raw_frame=frame,
                deadzone=args.deadzone
            )

            # FPS
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                display_fps = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()

            method_tag = "refined" if args.refine else "bbox"
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
                print(f"[+] Saved: {fname}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Headless mode (with guidance)
# ---------------------------------------------------------------------------
def run_headless(args):
    print("[*] Loading X detection model...")
    model = load_model(args.weights)

    mode = MODES[args.mode]
    pipeline = build_gstreamer_pipeline(**mode)
    infer_w = mode["display_width"]
    infer_h = mode["display_height"]
    print(f"[*] Opening camera: {args.mode} → {infer_w}x{infer_h}")
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

            # inference
            results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

            # find best detection
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
                # find X center
                if args.refine:
                    cx, cy = find_x_center_refined(frame, best_box)
                else:
                    cx, cy = find_x_center_bbox(best_box)

                direction_str, dx, dy, centered = compute_guidance(
                    cx, cy, frame_w, frame_h, args.deadzone
                )

                cls_name = best_result.names[int(best_box.cls[0])]
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])

                if centered:
                    print(f"\r[✓ CENTERED] {cls_name} conf={best_conf:.2f} "
                          f"X@({cx},{cy}) offset=({dx:+d},{dy:+d}) "
                          f"FPS={display_fps:.1f}    ")
                else:
                    print(f"\r[MOVE] {direction_str}  |  "
                          f"{cls_name} conf={best_conf:.2f} "
                          f"X@({cx},{cy}) offset=({dx:+d},{dy:+d}) "
                          f"FPS={display_fps:.1f}    ")
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
# Snapshot mode (with guidance)
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

    frame_h, frame_w = frame.shape[:2]
    results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

    # annotate
    annotated = frame.copy()

    # draw frame center crosshair
    mid_x, mid_y = frame_w // 2, frame_h // 2
    cv2.line(annotated, (mid_x - 30, mid_y), (mid_x + 30, mid_y), (128, 128, 128), 2)
    cv2.line(annotated, (mid_x, mid_y - 30), (mid_x, mid_y + 30), (128, 128, 128), 2)

    det_count = 0
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < args.conf:
                continue
            det_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = result.names[int(box.cls[0])]

            # draw bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # find center
            if args.refine:
                cx, cy = find_x_center_refined(frame, box)
            else:
                cx, cy = find_x_center_bbox(box)

            direction_str, dx, dy, centered = compute_guidance(
                cx, cy, frame_w, frame_h, args.deadzone
            )

            # draw center dot + line to frame center
            color = (0, 255, 0) if centered else (0, 0, 255)
            cv2.circle(annotated, (cx, cy), 12, color, -1)
            cv2.circle(annotated, (cx, cy), 14, (255, 255, 255), 3)
            cv2.line(annotated, (cx, cy), (mid_x, mid_y), color, 2)

            # label
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            # guidance on image
            cv2.putText(annotated, direction_str, (30, frame_h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 0) if centered else (0, 165, 255), 3)

            # terminal output
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

    # new guidance options
    parser.add_argument("--refine", action="store_true",
                        help="Use green-segmentation within bbox for precise "
                             "X center (default: bbox midpoint)")
    parser.add_argument("--deadzone", type=int, default=50,
                        help="Pixel deadzone for 'centered' (default: 50)")

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
