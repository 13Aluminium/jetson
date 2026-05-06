#!/usr/bin/env python3
"""
Script 5B: LAND ON X — Competition Version (Large 5m Green X)
================================================================
Two-phase CV approach:
    Phase 1 (YOLO, above SWITCH_ALT ~4m):
        Standard YOLO detection — whole X visible, bbox center = true center.
        Get well-centered before transitioning.

    Phase 2 (Line Intersection CV, below SWITCH_ALT):
        X overflows the frame. YOLO bbox center becomes unreliable.
        Instead: HSV green mask → skeletonize → find intersection point
        of the two X arms → navigate to that intersection.

State machine:
    TAKEOFF → SEARCH → ACQUIRE → DESCEND → [SWITCH to Phase 2] →
    ACQUIRE_CV → DESCEND_CV → FINAL_CV → LAND

Terminal 1: mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 --out=udp:127.0.0.1:14551
Terminal 2: python3 5b_land_on_x_competition.py

    SITL:   python3 5b_land_on_x_competition.py --sitl
    Dry:    python3 5b_land_on_x_competition.py --dry-run

Failsafes:
    Ctrl+C → RTL | Exception → RTL | Lost target 10s → RTL | Search timeout 60s → RTL
"""

import argparse, time, math, os, cv2
import numpy as np
from datetime import datetime
from flight_utils import (FlightController, SafeFlight, open_camera,
                          load_yolo, detect_x, pixels_to_meters,
                          get_camera_fps,
                          TAKEOFF_ALT, FRAME_W, FRAME_H,
                          confirm, create_log, log)

# ── Phase transition altitude ─────────────────────────────────
# Below this altitude, switch from YOLO to line-intersection CV.
# At 4m with 73° HFOV, ground width ≈ 5.8m — the 5m X is still
# mostly visible (YOLO works) AND the arms are thick enough for
# line detection. This gives overlap where both methods work.
SWITCH_ALT = 4.0

# ── YOLO phase parameters (Phase 1, above SWITCH_ALT) ────────
DESCEND_STEP     = 1.0       # meters per descent step
DEADZONE_HIGH    = 60        # px — centered threshold at high alt
SPEED_HIGH       = 0.3       # m/s — centering speed at high alt

# ── Tighter centering before switch ──────────────────────────
# Before we hand off to Phase 2, we want to be VERY well centered
# so that at lower altitude the intersection is in-frame.
SWITCH_DEADZONE  = 30        # px — must be this centered before switching
SWITCH_CONFIRM   = 5         # consecutive centered frames before switching

# ── CV phase parameters (Phase 2, below SWITCH_ALT) ──────────
DESCEND_STEP_CV  = 0.5       # meters per descent step (smaller = safer)
FINAL_ALT        = 1.5       # switch to final approach below this
LAND_ALT         = 0.8       # trigger LAND below this
DEADZONE_CV      = 25        # px — centered threshold for CV phase
SPEED_CV         = 0.15      # m/s — centering speed (slow = precise)
DESCENT_VZ       = 0.3       # m/s — descent rate
DESCENT_VZ_FINAL = 0.15      # m/s — slower descent in final approach

# ── Common parameters ─────────────────────────────────────────
LOST_TIMEOUT     = 10.0      # seconds without detection → RTL
SEARCH_TIMEOUT   = 60.0      # seconds searching at start → RTL
VEL_RATE         = 0.2       # seconds between velocity commands

# ── Green color thresholds (HSV) ──────────────────────────────
# These define what "green" looks like to the camera. You may need
# to tune these on-site depending on lighting and the specific
# green of your X. Print a test frame and check HSV values.
#
# Hue: 35-85 covers most greens (OpenCV hue range is 0-179).
# Saturation: >40 excludes washed-out / gray areas.
# Value: >40 excludes very dark areas.
GREEN_HSV_LOW  = np.array([35, 40, 40])
GREEN_HSV_HIGH = np.array([85, 255, 255])

# Minimum green pixel fraction to consider the mask valid.
# At low alt over a 5m X, green should be a big chunk of the frame.
MIN_GREEN_FRACTION = 0.03   # 3% of frame pixels must be green

# ── Video / overlay ───────────────────────────────────────────
OVERLAY_FONT       = cv2.FONT_HERSHEY_SIMPLEX
OVERLAY_COLOR_OK   = (0, 255, 0)
OVERLAY_COLOR_LOST = (0, 0, 255)
OVERLAY_COLOR_CENTER = (0, 255, 255)
OVERLAY_COLOR_CV   = (255, 165, 0)   # orange — for CV phase markers


# ===========================================================================
# GREEN X LINE-INTERSECTION DETECTION (Phase 2)
# ===========================================================================

def detect_green_mask(frame):
    """
    Segment green regions from the frame using HSV thresholding.
    Returns the binary mask and the green pixel fraction.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_HSV_LOW, GREEN_HSV_HIGH)

    # Morphological cleanup:
    # 1. Close — fill small gaps within the X arms
    # 2. Open — remove small noise blobs
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    fraction = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])
    return mask, fraction


def find_intersection_skeleton(mask):
    """
    Skeletonize the green mask and find the intersection point
    (the point where the two X arms cross).

    The intersection point in a skeleton has >2 neighbors — it's
    where branches meet. We find the point with the highest
    connectivity (most skeleton neighbors in a 3×3 window).

    Returns (cx, cy) in pixel coords, or None if not found.
    """
    # Skeletonize: thin the mask to 1-pixel-wide lines
    # We use iterative morphological thinning.
    skeleton = _skeletonize(mask)

    if np.count_nonzero(skeleton) < 20:
        return None, skeleton

    # For each skeleton pixel, count its 8-connected neighbors.
    # The intersection point has the most neighbors (typically 3 or 4).
    # A normal line point has exactly 2 neighbors.
    # An endpoint has 1 neighbor.
    #
    # We convolve with a 3×3 kernel of ones (minus center) to count neighbors.
    skel_f = (skeleton > 0).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1, 1] = 0
    neighbor_count = cv2.filter2D(skel_f, -1, kernel)

    # Mask to only skeleton pixels
    neighbor_count = neighbor_count * skel_f

    # Junction points: skeleton pixels with >2 neighbors
    junction_mask = (neighbor_count > 2) & (skel_f > 0)
    junctions = np.where(junction_mask)

    if len(junctions[0]) == 0:
        # No clear junction — fall back to skeleton centroid
        skel_points = np.where(skeleton > 0)
        if len(skel_points[0]) == 0:
            return None, skeleton
        cy = int(np.mean(skel_points[0]))
        cx = int(np.mean(skel_points[1]))
        return (cx, cy), skeleton

    # Multiple junction pixels — cluster them and take the centroid
    # of the largest cluster (the true intersection region).
    # Junction pixels near the intersection form a small cluster.
    junction_points = np.column_stack((junctions[1], junctions[0]))  # (x, y)

    if len(junction_points) == 1:
        return (int(junction_points[0][0]), int(junction_points[0][1])), skeleton

    # Simple clustering: find the densest region of junction points.
    # Use the centroid of all junction points — for an X, the junction
    # region is compact and centered on the true intersection.
    cx = int(np.mean(junction_points[:, 0]))
    cy = int(np.mean(junction_points[:, 1]))

    return (cx, cy), skeleton


def find_intersection_hough(mask):
    """
    Fallback: use HoughLinesP to find the two dominant line directions,
    then compute their intersection analytically.

    Returns (cx, cy) in pixel coords, or None if not found.
    """
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=80, maxLineGap=30)

    if lines is None or len(lines) < 2:
        return None

    # Cluster lines by angle into two groups (the two arms of the X).
    # Compute angle of each line segment.
    angles = []
    line_list = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.atan2(y2 - y1, x2 - x1)
        # Normalize to [0, π) — direction doesn't matter
        if angle < 0:
            angle += math.pi
        angles.append(angle)
        line_list.append((x1, y1, x2, y2))

    angles = np.array(angles)

    # Simple 2-means clustering on angles.
    # Start with the first line's angle as cluster 1 seed.
    # Find the first line whose angle differs by >30° as cluster 2 seed.
    seed1 = angles[0]
    seed2 = None
    for a in angles[1:]:
        diff = abs(a - seed1)
        diff = min(diff, math.pi - diff)  # handle wrap-around
        if diff > math.radians(30):
            seed2 = a
            break

    if seed2 is None:
        # All lines roughly parallel — can't find intersection
        return None

    # Assign each line to the closer cluster
    group1 = []
    group2 = []
    for i, a in enumerate(angles):
        d1 = abs(a - seed1)
        d1 = min(d1, math.pi - d1)
        d2 = abs(a - seed2)
        d2 = min(d2, math.pi - d2)
        if d1 < d2:
            group1.append(line_list[i])
        else:
            group2.append(line_list[i])

    if len(group1) == 0 or len(group2) == 0:
        return None

    # Fit a representative line for each group (average the endpoints).
    def avg_line(group):
        """Compute a representative line from a group of segments."""
        # Use all segment midpoints and the average direction.
        mid_xs, mid_ys = [], []
        dx_sum, dy_sum = 0.0, 0.0
        for x1, y1, x2, y2 in group:
            mid_xs.append((x1 + x2) / 2)
            mid_ys.append((y1 + y2) / 2)
            ddx, ddy = x2 - x1, y2 - y1
            length = math.sqrt(ddx**2 + ddy**2)
            if length > 0:
                dx_sum += ddx / length
                dy_sum += ddy / length
        mx = np.mean(mid_xs)
        my = np.mean(mid_ys)
        return mx, my, dx_sum, dy_sum

    mx1, my1, dx1, dy1 = avg_line(group1)
    mx2, my2, dx2, dy2 = avg_line(group2)

    # Find intersection of two lines:
    # Line 1: P1 + t * D1 where P1 = (mx1, my1), D1 = (dx1, dy1)
    # Line 2: P2 + s * D2 where P2 = (mx2, my2), D2 = (dx2, dy2)
    # Solve: P1 + t*D1 = P2 + s*D2
    #   dx1*t - dx2*s = mx2 - mx1
    #   dy1*t - dy2*s = my2 - my1
    det = dx1 * (-dy2) - (-dx2) * dy1
    if abs(det) < 1e-6:
        return None  # parallel lines

    t = ((mx2 - mx1) * (-dy2) - (my2 - my1) * (-dx2)) / det
    ix = mx1 + t * dx1
    iy = my1 + t * dy1

    return (int(round(ix)), int(round(iy)))


def detect_x_center_cv(frame):
    """
    Detect the center of a large green X using color segmentation
    and line intersection. Works when the X overflows the frame.

    Returns a dict similar to detect_x():
        {'cx': int, 'cy': int, 'conf': float, 'method': str,
         'mask_fraction': float}
    or None if detection fails.
    """
    mask, fraction = detect_green_mask(frame)

    if fraction < MIN_GREEN_FRACTION:
        return None

    # Method 1: Skeleton-based intersection detection
    result, skeleton = find_intersection_skeleton(mask)
    if result is not None:
        cx, cy = result
        # Sanity check: intersection should be within a reasonable
        # range of the frame (can be slightly outside if off-center,
        # but not wildly off).
        h, w = frame.shape[:2]
        if -w < cx < 2 * w and -h < cy < 2 * h:
            # Confidence based on green fraction (more green = more confident)
            conf = min(fraction / 0.15, 1.0)  # 15%+ green → full confidence
            return {
                'cx': cx, 'cy': cy,
                'conf': conf,
                'method': 'skeleton',
                'mask_fraction': fraction,
                'mask': mask,
                'skeleton': skeleton,
            }

    # Method 2: Hough line intersection (fallback)
    result_h = find_intersection_hough(mask)
    if result_h is not None:
        cx, cy = result_h
        h, w = frame.shape[:2]
        if -w < cx < 2 * w and -h < cy < 2 * h:
            conf = min(fraction / 0.15, 1.0) * 0.8  # slightly lower confidence
            return {
                'cx': cx, 'cy': cy,
                'conf': conf,
                'method': 'hough',
                'mask_fraction': fraction,
                'mask': mask,
                'skeleton': skeleton,
            }

    # Method 3: Mask centroid (last resort — less accurate but still useful)
    # The centroid of all green pixels gives a rough estimate of direction.
    green_points = np.where(mask > 0)
    if len(green_points[0]) > 100:
        cy = int(np.mean(green_points[0]))
        cx = int(np.mean(green_points[1]))
        conf = min(fraction / 0.15, 1.0) * 0.5  # lowest confidence
        return {
            'cx': cx, 'cy': cy,
            'conf': conf,
            'method': 'centroid',
            'mask_fraction': fraction,
            'mask': mask,
            'skeleton': skeleton,
        }

    return None


def _skeletonize(mask):
    """
    Morphological skeletonization (Zhang-Suen style thinning).
    Produces a 1-pixel-wide skeleton of the binary mask.
    """
    img = (mask > 0).astype(np.uint8)
    skeleton = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        dilated = cv2.dilate(eroded, element)
        diff = cv2.subtract(img, dilated)
        skeleton = cv2.bitwise_or(skeleton, diff)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return skeleton * 255


# ===========================================================================
# VIDEO OVERLAY (enhanced for Phase 2)
# ===========================================================================

def draw_overlay(frame, state, det, cur_alt, fc, centered=False,
                 cv_det=None, show_mask=False):
    """
    Draw detection overlay, crosshair, and HUD.
    For Phase 2 states, also show the CV detection markers.
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Crosshair at frame center
    size = 30
    cv2.line(frame, (cx - size, cy), (cx + size, cy), OVERLAY_COLOR_CENTER, 1)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), OVERLAY_COLOR_CENTER, 1)

    # Deadzone circle (adaptive)
    if state in ("ACQUIRE_CV", "DESCEND_CV", "FINAL_CV"):
        dz = DEADZONE_CV
    elif cur_alt < FINAL_ALT + 1:
        dz = SWITCH_DEADZONE
    else:
        dz = DEADZONE_HIGH
    cv2.circle(frame, (cx, cy), dz, OVERLAY_COLOR_CENTER, 1)

    # ── YOLO detection (Phase 1) ──────────────────────────────
    if det and not cv_det:
        x1, y1, x2, y2 = det['bbox']
        color = OVERLAY_COLOR_OK
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        dcx, dcy = int(det['cx']), int(det['cy'])
        cv2.line(frame, (cx, cy), (dcx, dcy), color, 1)
        label = f"X {det['conf']:.0%}"
        cv2.putText(frame, label, (int(x1), int(y1) - 8),
                    OVERLAY_FONT, 0.6, color, 2)
        dx_px = det['cx'] - cx
        dy_px = det['cy'] - cy
        cv2.putText(frame, f"dx={dx_px:+.0f} dy={dy_px:+.0f}px",
                    (int(x1), int(y2) + 20), OVERLAY_FONT, 0.5, color, 1)

    # ── CV detection (Phase 2) ────────────────────────────────
    elif cv_det:
        dcx, dcy = int(cv_det['cx']), int(cv_det['cy'])
        color = OVERLAY_COLOR_CV

        # Draw detected intersection point
        cv2.drawMarker(frame, (dcx, dcy), color, cv2.MARKER_CROSS, 40, 2)
        cv2.circle(frame, (dcx, dcy), 15, color, 2)

        # Line from frame center to intersection
        cv2.line(frame, (cx, cy), (dcx, dcy), color, 2)

        # Label
        method = cv_det.get('method', '?')
        conf = cv_det.get('conf', 0)
        frac = cv_det.get('mask_fraction', 0)
        label = f"CV:{method} {conf:.0%} grn={frac:.1%}"
        cv2.putText(frame, label, (10, h - 40), OVERLAY_FONT, 0.55, color, 2)

        dx_px = dcx - cx
        dy_px = dcy - cy
        cv2.putText(frame, f"dx={dx_px:+.0f} dy={dy_px:+.0f}px",
                    (10, h - 15), OVERLAY_FONT, 0.5, color, 1)

        # Optionally overlay the green mask semi-transparent
        if show_mask and 'mask' in cv_det:
            mask_rgb = np.zeros_like(frame)
            mask_rgb[:, :, 1] = cv_det['mask']  # green channel
            frame = cv2.addWeighted(frame, 1.0, mask_rgb, 0.3, 0)

    elif state not in ("TAKEOFF", "NAVIGATE", "SETTLING"):
        cv2.putText(frame, "NO TARGET", (cx - 60, cy + 50),
                    OVERLAY_FONT, 0.8, OVERLAY_COLOR_LOST, 2)

    # ── Phase indicator ───────────────────────────────────────
    phase_label = "YOLO" if state in ("SEARCH", "ACQUIRE", "DESCEND") else "CV"
    if state in ("ACQUIRE_CV", "DESCEND_CV", "FINAL_CV"):
        phase_label = "CV"
    cv2.putText(frame, f"PHASE: {phase_label}", (w - 200, 25),
                OVERLAY_FONT, 0.6, OVERLAY_COLOR_CV if phase_label == "CV"
                else OVERLAY_COLOR_OK, 2)

    # ── HUD ───────────────────────────────────────────────────
    hud_color = OVERLAY_COLOR_OK if (det or cv_det) else OVERLAY_COLOR_LOST
    lines = [
        f"STATE: {state}",
        f"ALT: {cur_alt:.1f}m",
        f"GPS: {fc.lat:.6f}, {fc.lon:.6f}",
        f"SATS: {fc.satellites}  FIX: {fc.gps_fix}",
        f"BATT: {fc.battery_pct}%",
        f"HDG: {fc.heading:.0f} deg",
    ]
    if centered:
        lines.append("** CENTERED **")
    if cur_alt <= SWITCH_ALT and state not in ("TAKEOFF", "SEARCH"):
        lines.append(f"SWITCH_ALT: {SWITCH_ALT}m")

    y_off = 25
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (11, y_off + i * 24),
                    OVERLAY_FONT, 0.55, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, y_off + i * 24),
                    OVERLAY_FONT, 0.55, hud_color, 1)

    # Timestamp
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    cv2.putText(frame, ts, (w - 160, h - 12),
                OVERLAY_FONT, 0.5, (255, 255, 255), 1)

    return frame


# ===========================================================================
# MAIN
# ===========================================================================

def main(args):
    if not args.dry_run and not args.sitl:
        if not confirm("5b_land_on_x_competition.py — COMPETITION SCRIPT",
                       f"Takeoff {args.alt}m → Find X (YOLO) → Center → "
                       f"Switch to CV at {SWITCH_ALT}m → Precision descend → LAND ON X\n"
                       f"  Green HSV range: {GREEN_HSV_LOW} — {GREEN_HSV_HIGH}\n"
                       f"  Video recording: ON"):
            return

    model = load_yolo(args.weights, imgsz=args.imgsz)

    fc = FlightController()
    if not args.dry_run:
        fc.connect()
        if not args.sitl and not fc.preflight():
            fc.close(); return

    cap = open_camera(sitl=args.sitl)
    if not cap and not args.sitl:
        print("[!] No camera — cannot detect X.")
        fc.close(); return

    # ── Green calibration check ───────────────────────────────
    if cap and not args.sitl:
        print("\n[CAL] Quick green check — point camera at the X...")
        time.sleep(1)
        for _ in range(5):
            cap.read()  # flush
        ret, cal_frame = cap.read()
        if ret:
            mask, frac = detect_green_mask(cal_frame)
            print(f"[CAL] Green fraction: {frac:.1%} "
                  f"(need >{MIN_GREEN_FRACTION:.1%})")
            if frac < 0.01:
                print("[CAL] ⚠ Very little green detected!")
                print("[CAL]   You may need to tune GREEN_HSV_LOW/HIGH")
                print(f"[CAL]   Current: low={GREEN_HSV_LOW} high={GREEN_HSV_HIGH}")
            else:
                print("[CAL] ✓ Green detected — looks good")
        print()

    # ── Video writer setup ────────────────────────────────────
    vw = None
    video_path = None
    video_path_tmp = None
    actual_fps = 20.0
    frame_count = 0
    record_t0 = None

    if cap:
        actual_fps = get_camera_fps(cap, sitl=args.sitl)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path_tmp = f"landing_comp_{ts}_tmp.mp4"
        video_path = f"landing_comp_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(video_path_tmp, fourcc, actual_fps,
                             (FRAME_W, FRAME_H))
        if not vw.isOpened():
            print("[!] WARNING: Could not open video writer")
            vw = None
        else:
            print(f"[REC] Recording → {video_path}  ({actual_fps:.1f} FPS)")

    log_fname, log_f = create_log("landing_comp")
    log(log_f, "COMPETITION X LANDING — Two-Phase CV")
    log(log_f, f"Alt={args.alt}m | Switch={SWITCH_ALT}m | "
               f"Final={FINAL_ALT}m | Land={LAND_ALT}m")
    log(log_f, f"Green HSV: {GREEN_HSV_LOW} — {GREEN_HSV_HIGH}")
    if video_path:
        log(log_f, f"Video: {video_path}")

    with SafeFlight(fc, camera=cap, video_writer=vw) as sf:

        state = "TAKEOFF"
        last_seen = 0          # last time any target was seen (YOLO or CV)
        search_t0 = 0          # when search started
        descend_tgt = 0        # target alt for current descent
        switch_count = 0       # consecutive centered frames before switch

        if args.dry_run:
            state = "SEARCH"
            log(log_f, "DRY RUN — skipping takeoff")

        # ══════════════════════════════════════════════════════
        # STATE MACHINE
        # ══════════════════════════════════════════════════════
        while state not in ("DONE", "ABORT"):
            if not args.dry_run:
                fc.poll()
            cur_alt = fc.alt if (not args.dry_run and fc.alt > 0.3) else args.alt

            # Read camera frame
            frame = None
            if cap:
                ret, frame = cap.read()
                if not ret:
                    frame = None

            # ── TAKEOFF ──────────────────────────────────────
            if state == "TAKEOFF":
                log(log_f, f"TAKEOFF → {args.alt}m")
                record_t0 = time.time()

                if not fc.set_guided():
                    state = "ABORT"; continue
                if not fc.arm():
                    state = "ABORT"; continue
                if not fc.takeoff(args.alt):
                    fc.set_rtl(); state = "ABORT"; continue
                if not fc.wait_alt(args.alt):
                    fc.set_rtl(); state = "ABORT"; continue

                log(log_f, "At altitude — stabilizing 3s")
                t0 = time.time()
                while time.time() - t0 < 3:
                    if cap:
                        ret, frm = cap.read()
                        if ret:
                            fc.poll()
                            det = detect_x(frm, model, args.conf, args.imgsz)
                            overlay = draw_overlay(frm.copy(), "TAKEOFF", det,
                                                   fc.alt, fc)
                            if vw:
                                vw.write(overlay); frame_count += 1
                    time.sleep(0.05)

                state = "SEARCH"
                search_t0 = time.time()
                continue

            # ── Write video frame (all states except TAKEOFF) ─
            # Some states (DESCEND, FINAL_CV, LAND) handle their
            # own recording inside their loops. For the others,
            # we record here.
            wrote_frame = False

            # ── SEARCH (YOLO) ────────────────────────────────
            if state == "SEARCH":
                if search_t0 == 0:
                    search_t0 = time.time()
                elapsed = time.time() - search_t0

                det = None
                if frame is not None:
                    det = detect_x(frame, model, args.conf, args.imgsz)

                if frame is not None and vw:
                    overlay = draw_overlay(frame.copy(), state, det, cur_alt, fc)
                    vw.write(overlay); frame_count += 1
                    wrote_frame = True

                if det:
                    log(log_f, f"X FOUND conf={det['conf']:.2f} "
                               f"@({det['cx']},{det['cy']})")
                    state = "ACQUIRE"
                    last_seen = time.time()
                    continue

                if elapsed > SEARCH_TIMEOUT:
                    log(log_f, f"SEARCH TIMEOUT ({SEARCH_TIMEOUT}s) → RTL")
                    if not args.dry_run:
                        fc.set_rtl()
                    state = "ABORT"; continue

                print(f"\r  [SEARCH] {elapsed:.0f}s / {SEARCH_TIMEOUT:.0f}s | "
                      f"Alt={cur_alt:.1f}m   ", end="", flush=True)
                time.sleep(0.05)

            # ── ACQUIRE (YOLO — Phase 1) ─────────────────────
            elif state == "ACQUIRE":
                det = None
                if frame is not None:
                    det = detect_x(frame, model, args.conf, args.imgsz)

                if frame is not None and vw:
                    overlay = draw_overlay(frame.copy(), state, det, cur_alt, fc)
                    vw.write(overlay); frame_count += 1
                    wrote_frame = True

                if det is None:
                    lost = time.time() - last_seen
                    if lost > LOST_TIMEOUT:
                        log(log_f, f"LOST X {lost:.0f}s → RTL")
                        if not args.dry_run:
                            fc.stop(); fc.set_rtl()
                        state = "ABORT"; continue
                    if not args.dry_run:
                        fc.stop()
                    print(f"\r  [ACQUIRE] Lost X — holding ({lost:.1f}s / "
                          f"{LOST_TIMEOUT:.0f}s)   ", end="", flush=True)
                    switch_count = 0
                    time.sleep(VEL_RATE)
                    continue

                last_seen = time.time()
                dx_px = det['cx'] - FRAME_W // 2
                dy_px = det['cy'] - FRAME_H // 2

                # Check if we should switch to CV phase
                # Condition: below SWITCH_ALT AND well-centered
                if cur_alt <= SWITCH_ALT + 0.5:
                    if abs(dx_px) <= SWITCH_DEADZONE and abs(dy_px) <= SWITCH_DEADZONE:
                        switch_count += 1
                        if switch_count >= SWITCH_CONFIRM:
                            log(log_f, f"SWITCH to CV phase at {cur_alt:.1f}m "
                                       f"(centered {switch_count}x)")
                            if not args.dry_run:
                                fc.stop()
                            state = "ACQUIRE_CV"
                            continue
                    else:
                        switch_count = 0

                # Normal centering
                if abs(dx_px) <= DEADZONE_HIGH and abs(dy_px) <= DEADZONE_HIGH:
                    log(log_f, f"CENTERED at {cur_alt:.1f}m "
                               f"({dx_px:+d},{dy_px:+d})px")
                    if not args.dry_run:
                        fc.stop()

                    # Re-draw with centered flag
                    if frame is not None and vw:
                        overlay = draw_overlay(frame.copy(), state, det,
                                               cur_alt, fc, centered=True)
                        vw.write(overlay); frame_count += 1

                    state = "DESCEND"
                    descend_tgt = max(cur_alt - DESCEND_STEP, SWITCH_ALT)
                    time.sleep(0.5)
                    continue

                # Compute correction
                m_fwd, m_right = pixels_to_meters(dx_px, dy_px, cur_alt)
                dist = math.sqrt(m_fwd**2 + m_right**2)
                scale = min(SPEED_HIGH / dist, 1.0) if dist > SPEED_HIGH else 0.5
                vx = m_fwd * scale
                vy = m_right * scale

                if not args.dry_run:
                    fc.velocity_body(vx, vy, 0)

                parts = []
                if abs(m_fwd) > 0.05:
                    parts.append(f"{'FWD' if m_fwd > 0 else 'BACK'} {abs(m_fwd):.2f}m")
                if abs(m_right) > 0.05:
                    parts.append(f"{'RIGHT' if m_right > 0 else 'LEFT'} {abs(m_right):.2f}m")
                print(f"\r  [ACQUIRE] {' + '.join(parts) or '~'} | "
                      f"v=({vx:.2f},{vy:.2f}) | Alt={cur_alt:.1f}m | "
                      f"conf={det['conf']:.2f} | sw={switch_count}  ",
                      end="", flush=True)
                time.sleep(VEL_RATE)

            # ── DESCEND (YOLO — Phase 1) ─────────────────────
            elif state == "DESCEND":
                log(log_f, f"DESCEND {cur_alt:.1f}m → {descend_tgt:.1f}m")
                t0 = time.time()
                while True:
                    if not args.dry_run:
                        fc.poll()
                    cur_alt = fc.alt if not args.dry_run else descend_tgt
                    if cur_alt <= descend_tgt + 0.3:
                        break
                    if time.time() - t0 > 15:
                        break
                    if not args.dry_run:
                        fc.velocity_ned(0, 0, DESCENT_VZ)

                    if cap:
                        ret, frm = cap.read()
                        if ret:
                            d = detect_x(frm, model, args.conf, args.imgsz)
                            if d:
                                last_seen = time.time()
                            overlay = draw_overlay(frm.copy(), "DESCEND", d,
                                                   cur_alt, fc)
                            if vw:
                                vw.write(overlay); frame_count += 1

                    print(f"\r  [DESCEND] {cur_alt:.1f}m → {descend_tgt:.1f}m   ",
                          end="", flush=True)
                    time.sleep(VEL_RATE)

                if not args.dry_run:
                    fc.stop()
                log(log_f, f"At {cur_alt:.1f}m — re-acquiring")
                time.sleep(1)
                state = "ACQUIRE"

            # ══════════════════════════════════════════════════
            # PHASE 2: CV-BASED (below SWITCH_ALT)
            # ══════════════════════════════════════════════════

            # ── ACQUIRE_CV (center using line intersection) ──
            elif state == "ACQUIRE_CV":
                cv_det = None
                if frame is not None:
                    cv_det = detect_x_center_cv(frame)

                if frame is not None and vw:
                    overlay = draw_overlay(frame.copy(), state, None,
                                           cur_alt, fc, cv_det=cv_det,
                                           show_mask=args.show_mask)
                    vw.write(overlay); frame_count += 1
                    wrote_frame = True

                if cv_det is None:
                    lost = time.time() - last_seen
                    if lost > LOST_TIMEOUT:
                        log(log_f, f"CV LOST target {lost:.0f}s → RTL")
                        if not args.dry_run:
                            fc.stop(); fc.set_rtl()
                        state = "ABORT"; continue
                    if not args.dry_run:
                        fc.stop()
                    print(f"\r  [ACQUIRE_CV] No green — holding ({lost:.1f}s / "
                          f"{LOST_TIMEOUT:.0f}s)   ", end="", flush=True)
                    time.sleep(VEL_RATE)
                    continue

                last_seen = time.time()
                dx_px = cv_det['cx'] - FRAME_W // 2
                dy_px = cv_det['cy'] - FRAME_H // 2

                if abs(dx_px) <= DEADZONE_CV and abs(dy_px) <= DEADZONE_CV:
                    # CENTERED via CV
                    log(log_f, f"CV CENTERED at {cur_alt:.1f}m "
                               f"({dx_px:+d},{dy_px:+d})px "
                               f"method={cv_det['method']} "
                               f"conf={cv_det['conf']:.2f}")
                    if not args.dry_run:
                        fc.stop()

                    if frame is not None and vw:
                        overlay = draw_overlay(frame.copy(), state, None,
                                               cur_alt, fc, centered=True,
                                               cv_det=cv_det)
                        vw.write(overlay); frame_count += 1

                    if cur_alt <= LAND_ALT + 0.5:
                        state = "LAND"
                    elif cur_alt <= FINAL_ALT + 0.5:
                        state = "FINAL_CV"
                    else:
                        state = "DESCEND_CV"
                        descend_tgt = max(cur_alt - DESCEND_STEP_CV, FINAL_ALT)
                    time.sleep(0.3)
                    continue

                # Compute correction
                m_fwd, m_right = pixels_to_meters(dx_px, dy_px, cur_alt)
                dist = math.sqrt(m_fwd**2 + m_right**2)
                scale = min(SPEED_CV / dist, 1.0) if dist > SPEED_CV else 0.4
                vx = m_fwd * scale
                vy = m_right * scale

                if not args.dry_run:
                    fc.velocity_body(vx, vy, 0)

                parts = []
                if abs(m_fwd) > 0.03:
                    parts.append(f"{'FWD' if m_fwd > 0 else 'BACK'} {abs(m_fwd):.2f}m")
                if abs(m_right) > 0.03:
                    parts.append(f"{'RIGHT' if m_right > 0 else 'LEFT'} {abs(m_right):.2f}m")
                print(f"\r  [ACQUIRE_CV] {' + '.join(parts) or '~'} | "
                      f"method={cv_det['method']} conf={cv_det['conf']:.2f} | "
                      f"Alt={cur_alt:.1f}m   ", end="", flush=True)
                time.sleep(VEL_RATE)

            # ── DESCEND_CV ───────────────────────────────────
            elif state == "DESCEND_CV":
                log(log_f, f"DESCEND_CV {cur_alt:.1f}m → {descend_tgt:.1f}m")
                t0 = time.time()
                while True:
                    if not args.dry_run:
                        fc.poll()
                    cur_alt = fc.alt if not args.dry_run else descend_tgt
                    if cur_alt <= descend_tgt + 0.3:
                        break
                    if time.time() - t0 > 15:
                        break
                    if not args.dry_run:
                        fc.velocity_ned(0, 0, DESCENT_VZ)

                    if cap:
                        ret, frm = cap.read()
                        if ret:
                            cv_d = detect_x_center_cv(frm)
                            if cv_d:
                                last_seen = time.time()
                            overlay = draw_overlay(frm.copy(), "DESCEND_CV",
                                                   None, cur_alt, fc,
                                                   cv_det=cv_d)
                            if vw:
                                vw.write(overlay); frame_count += 1

                    print(f"\r  [DESCEND_CV] {cur_alt:.1f}m → {descend_tgt:.1f}m   ",
                          end="", flush=True)
                    time.sleep(VEL_RATE)

                if not args.dry_run:
                    fc.stop()
                log(log_f, f"At {cur_alt:.1f}m — re-acquiring (CV)")
                time.sleep(0.8)
                state = "ACQUIRE_CV"

            # ── FINAL_CV (simultaneous centering + slow descent) ─
            elif state == "FINAL_CV":
                log(log_f, f"FINAL_CV at {cur_alt:.1f}m")
                t0 = time.time()
                while True:
                    if not args.dry_run:
                        fc.poll()
                    cur_alt = fc.alt if not args.dry_run else LAND_ALT

                    cv_det = None
                    if cap:
                        ret, frm = cap.read()
                        if ret:
                            frame = frm
                            cv_det = detect_x_center_cv(frm)

                    if frame is not None and vw:
                        overlay = draw_overlay(frame.copy(), "FINAL_CV",
                                               None, cur_alt, fc, cv_det=cv_det)
                        vw.write(overlay); frame_count += 1

                    if cur_alt <= LAND_ALT + 0.3:
                        log(log_f, f"Below {LAND_ALT}m → LAND")
                        state = "LAND"; break
                    if time.time() - t0 > 30:
                        log(log_f, "Final CV timeout → LAND")
                        state = "LAND"; break

                    if cv_det:
                        last_seen = time.time()
                        dx_px = cv_det['cx'] - FRAME_W // 2
                        dy_px = cv_det['cy'] - FRAME_H // 2
                        m_fwd, m_right = pixels_to_meters(dx_px, dy_px, cur_alt)
                        dist = math.sqrt(m_fwd**2 + m_right**2)
                        sc = min(SPEED_CV / dist, 1.0) if dist > SPEED_CV else 0.3
                        vx, vy = m_fwd * sc, m_right * sc
                        if not args.dry_run:
                            fc.velocity_body(vx, vy, DESCENT_VZ_FINAL)
                        print(f"\r  [FINAL_CV] Alt={cur_alt:.1f}m | "
                              f"({dx_px:+d},{dy_px:+d})px | "
                              f"v=({vx:.2f},{vy:.2f},{DESCENT_VZ_FINAL}) | "
                              f"{cv_det['method']}   ",
                              end="", flush=True)
                    else:
                        lost = time.time() - last_seen
                        if not args.dry_run:
                            fc.stop()
                        print(f"\r  [FINAL_CV] Lost ({lost:.1f}s) | "
                              f"Alt={cur_alt:.1f}m   ", end="", flush=True)
                        if lost > 5:
                            log(log_f, "Lost in final CV 5s → LAND anyway")
                            state = "LAND"; break
                    time.sleep(VEL_RATE)

            # ── LAND ─────────────────────────────────────────
            elif state == "LAND":
                log(log_f, f"LAND at {cur_alt:.1f}m")
                if not args.dry_run:
                    fc.set_land()
                    land_t0 = time.time()
                    while fc.armed and (time.time() - land_t0 < 30):
                        fc.poll()
                        if cap:
                            ret, frm = cap.read()
                            if ret:
                                cv_d = detect_x_center_cv(frm)
                                overlay = draw_overlay(frm.copy(), "LANDING",
                                                       None, fc.alt, fc,
                                                       cv_det=cv_d)
                                if vw:
                                    vw.write(overlay); frame_count += 1
                        time.sleep(0.1)

                log(log_f, "LANDED ON X!")
                print("\n\n" + "=" * 60)
                print("  ★ ★ ★  LANDED ON X!  ★ ★ ★")
                print("=" * 60 + "\n")
                state = "DONE"

            # ── ABORT ────────────────────────────────────────
            elif state == "ABORT":
                log(log_f, "ABORTED")
                if not args.dry_run:
                    fc.wait_disarmed(timeout=60)
                state = "DONE"

    # ── Finalize video ────────────────────────────────────────
    if vw:
        vw.release()
        record_elapsed = time.time() - record_t0 if record_t0 else 1
        measured_fps = frame_count / max(record_elapsed, 0.001)
        log(log_f, f"Video: {frame_count} frames, {record_elapsed:.1f}s, "
                   f"measured {measured_fps:.1f} FPS")

        if frame_count > 0 and os.path.isfile(video_path_tmp):
            try:
                import subprocess
                subprocess.run([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", video_path_tmp,
                    "-vf", f"setpts=N/{measured_fps:.2f}/TB",
                    "-r", f"{measured_fps:.2f}",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    video_path
                ], check=True)
                os.remove(video_path_tmp)
                log(log_f, f"Remuxed → {video_path} @ {measured_fps:.1f} FPS")
            except Exception as e:
                os.rename(video_path_tmp, video_path)
                log(log_f, f"ffmpeg remux failed ({e}), raw file kept: {video_path}")
        else:
            if os.path.isfile(video_path_tmp):
                os.rename(video_path_tmp, video_path)

    log_f.close()
    print(f"[*] Log:   {log_fname}")
    if video_path:
        print(f"[*] Video: {video_path}")
    print("[*] Done!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Competition X Landing — Two-Phase CV (YOLO + Green Line Intersection)")
    p.add_argument("--alt", type=float, default=TAKEOFF_ALT,
                   help=f"Takeoff altitude (default {TAKEOFF_ALT}m)")
    p.add_argument("--weights", default="best_22.pt",
                   help="YOLO weights file")
    p.add_argument("--conf", type=float, default=0.50,
                   help="YOLO confidence threshold")
    p.add_argument("--imgsz", type=int, default=640,
                   help="YOLO input size")
    p.add_argument("--show-mask", action="store_true",
                   help="Overlay green mask on video (debug)")
    p.add_argument("--dry-run", action="store_true",
                   help="Camera only, no flight (tests detection)")
    p.add_argument("--sitl", action="store_true",
                   help="SITL mode (webcam)")
    main(p.parse_args())
