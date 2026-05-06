#!/usr/bin/env python3
"""
Script 8: Fly to GPS Coordinates → Detect X → Land on X
==========================================================
Takeoff to ALT → stabilize 3s → fly to target GPS coords at 0.5 m/s →
hover 3s → search for X → center over X → descend in steps → land on X.

Combines the GPS navigation logic from 6_fly_to_coords.py with the
full X-landing state machine from 5_land_on_x.py.

Usage:
    Terminal 1: mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 --out=udp:127.0.0.1:14551
    Terminal 2: python3 8_fly_to_coords_and_land_on_x.py --lat 33.78310 --lon -118.10940

    With options:
    python3 8_fly_to_coords_and_land_on_x.py --lat 33.78310 --lon -118.10940 --alt 8 --speed 1.0

    SITL:
    Terminal 1: sim_vehicle.py -v ArduCopter --console --map --out=udp:127.0.0.1:14551
    Terminal 2: python3 8_fly_to_coords_and_land_on_x.py --lat -35.36320 --lon 149.16520 --sitl

Failsafes:
    Ctrl+C → RTL | Exception → RTL | X lost 10s → RTL | Search timeout 60s → RTL
    Nav timeout 300s → RTL
"""

import argparse, time, math, os, sys, cv2
from datetime import datetime
from pymavlink import mavutil

from flight_utils import (FlightController, SafeFlight, open_camera,
                          load_yolo, detect_x, pixels_to_meters,
                          get_camera_fps,
                          TAKEOFF_ALT, FRAME_W, FRAME_H,
                          confirm, create_log, log)

# ── GPS navigation parameters ────────────────────────────────
DEFAULT_SPEED  = 0.5     # m/s cruise speed
ARRIVE_RADIUS  = 1.5     # meters — "arrived" when this close
NAV_TIMEOUT    = 300     # seconds max for navigation (safety)
POLL_RATE      = 0.5     # seconds between position checks during nav
SETTLE_TIME    = 3.0     # seconds to hover/settle after takeoff & after arrival

# ── X-landing parameters (from script 5) ─────────────────────
DESCEND_STEP   = 1.0     # meters per step
FINAL_ALT      = 1.5     # switch to final approach below this
LAND_ALT       = 0.8     # trigger LAND below this

DEADZONE_HIGH  = 60      # px — centered at high altitude
DEADZONE_LOW   = 30      # px — centered near ground
SPEED_HIGH     = 0.3     # m/s — centering speed at altitude
SPEED_LOW      = 0.15    # m/s — centering speed near ground

LOST_TIMEOUT   = 10.0    # seconds lost → RTL
SEARCH_TIMEOUT = 60.0    # seconds searching → RTL
DESCENT_VZ     = 0.3     # m/s descent rate
VEL_RATE       = 0.2     # seconds between velocity commands

# ── Video / overlay ───────────────────────────────────────────
OVERLAY_FONT       = cv2.FONT_HERSHEY_SIMPLEX
OVERLAY_COLOR_OK   = (0, 255, 0)       # green — X detected
OVERLAY_COLOR_LOST = (0, 0, 255)       # red — X lost
OVERLAY_COLOR_CENTER = (0, 255, 255)   # yellow — crosshair


# ===========================================================================
# GPS HELPERS
# ===========================================================================
def haversine(lat1, lon1, lat2, lon2):
    """Distance in meters between two GPS points."""
    R = 6_371_000
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lat1, lon1, lat2, lon2):
    """Initial bearing in degrees from point 1 to point 2."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(rlat2)
    y = (math.cos(rlat1) * math.sin(rlat2) -
         math.sin(rlat1) * math.cos(rlat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def bearing_to_compass(deg):
    """Convert bearing degrees to compass direction."""
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(deg / 22.5) % 16
    return dirs[idx]


def send_goto(fc, lat, lon, alt):
    """
    Command the FC to fly to a global GPS coordinate in GUIDED mode.
    Uses SET_POSITION_TARGET_GLOBAL_INT.
    """
    fc.master.mav.set_position_target_global_int_send(
        0,                                              # time_boot_ms
        fc.master.target_system,
        fc.master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        0b0000_1111_1111_1000,  # type_mask: position only
        int(lat * 1e7),         # lat_int (degE7)
        int(lon * 1e7),         # lon_int (degE7)
        alt,                    # alt (meters, relative)
        0, 0, 0,                # vx, vy, vz (ignored)
        0, 0, 0,                # afx, afy, afz (ignored)
        0, 0                    # yaw, yaw_rate (ignored)
    )


def set_speed(fc, speed_mps):
    """Set the cruise speed via MAV_CMD_DO_CHANGE_SPEED."""
    fc.master.mav.command_long_send(
        fc.master.target_system,
        fc.master.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
        0,
        0,              # speed type: 0 = airspeed
        speed_mps,      # speed in m/s
        -1,             # throttle (no change)
        0, 0, 0, 0
    )


# ===========================================================================
# VIDEO OVERLAY
# ===========================================================================
def draw_overlay(frame, state, det, cur_alt, fc, centered=False):
    """Draw detection overlay, crosshair, and HUD onto the frame."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Crosshair at frame center
    size = 30
    cv2.line(frame, (cx - size, cy), (cx + size, cy), OVERLAY_COLOR_CENTER, 1)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), OVERLAY_COLOR_CENTER, 1)

    # Adaptive deadzone circle
    dz = DEADZONE_LOW if cur_alt < FINAL_ALT + 1 else DEADZONE_HIGH
    cv2.circle(frame, (cx, cy), dz, OVERLAY_COLOR_CENTER, 1)

    # Detection bbox + line to center
    if det:
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
    else:
        cv2.putText(frame, "NO X", (cx - 30, cy + 50),
                    OVERLAY_FONT, 0.8, OVERLAY_COLOR_LOST, 2)

    # HUD: state / altitude / GPS / battery
    hud_color = OVERLAY_COLOR_OK if det else OVERLAY_COLOR_LOST
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

    y_off = 25
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (11, y_off + i * 24),
                    OVERLAY_FONT, 0.55, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, y_off + i * 24),
                    OVERLAY_FONT, 0.55, hud_color, 1)

    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    cv2.putText(frame, ts, (w - 160, h - 12),
                OVERLAY_FONT, 0.5, (255, 255, 255), 1)
    return frame


def record_frame(cap, vw, model, state, fc, args, frame_count):
    """Read a frame, run detection, draw overlay, write to video. Returns (det, frame, frame_count)."""
    if not cap:
        return None, None, frame_count
    ret, frame = cap.read()
    if not ret:
        return None, None, frame_count
    det = detect_x(frame, model, args.conf, args.imgsz) if model else None
    if vw:
        overlay = draw_overlay(frame.copy(), state, det, fc.alt, fc)
        vw.write(overlay)
        frame_count += 1
    return det, frame, frame_count


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    args = build_parser().parse_args()
    target_lat = args.lat
    target_lon = args.lon

    # ── Connect & fetch current position ──────────────────────
    print("[NAV] Connecting to flight controller …")
    fc = FlightController()
    fc.connect()
    fc.poll()

    # Wait for GPS fix
    print("[NAV] Waiting for GPS fix …")
    t0 = time.time()
    while fc.gps_fix < 3:
        fc.poll()
        elapsed = time.time() - t0
        print(f"\r[NAV] GPS fix: {fc.gps_fix}  sats: {fc.satellites}  "
              f"({elapsed:.0f}s)  ", end="", flush=True)
        if elapsed > 60:
            print("\n[!] ERROR: Could not get GPS fix after 60s")
            fc.close()
            sys.exit(1)
        time.sleep(1)
    print()

    cur_lat = fc.lat
    cur_lon = fc.lon

    # ── Calculate distance & bearing ──────────────────────────
    dist = haversine(cur_lat, cur_lon, target_lat, target_lon)
    brng = bearing(cur_lat, cur_lon, target_lat, target_lon)
    compass = bearing_to_compass(brng)

    # ── Show flight plan ──────────────────────────────────────
    print()
    print("=" * 60)
    print("  FLY TO COORDS → DETECT X → LAND ON X")
    print("=" * 60)
    print()
    print(f"  CURRENT POSITION:")
    print(f"    Lat:  {cur_lat:.8f}")
    print(f"    Lon:  {cur_lon:.8f}")
    print(f"    Sats: {fc.satellites}   Fix: {fc.gps_fix}")
    print(f"    Batt: {fc.battery_pct}%")
    print()
    print(f"  TARGET POSITION:")
    print(f"    Lat:  {target_lat:.8f}")
    print(f"    Lon:  {target_lon:.8f}")
    print()
    print(f"  FLIGHT DETAILS:")
    print(f"    Distance:  {dist:.1f} m")
    print(f"    Bearing:   {brng:.1f}° ({compass})")
    print(f"    Altitude:  {args.alt} m")
    print(f"    Speed:     {args.speed} m/s")
    print(f"    Est. time: {dist / args.speed:.0f}s (cruise only)")
    print()
    print(f"  PLAN: Takeoff → Fly {dist:.1f}m {compass} → Hover 3s →")
    print(f"        Detect X → Center → Descend → Land on X")
    print("=" * 60)
    print()

    # ── Safety checks ─────────────────────────────────────────
    if dist > 500:
        print(f"[!] WARNING: Distance is {dist:.0f}m — that's over 500m!")
        print("    Double-check your coordinates.")
        print()

    if dist < 1:
        print(f"[!] Target is only {dist:.2f}m away — skipping GPS nav,")
        print(f"    will do X-landing from current position.")

    # ── Confirmation ──────────────────────────────────────────
    if not args.sitl:
        if not confirm("8_fly_to_coords_and_land_on_x.py",
                       f"Fly {dist:.1f}m {compass} at {args.alt}m → Hover 3s → "
                       f"Detect X → Center → Descend → LAND ON X\n"
                       f"  Video recording: ON"):
            fc.close()
            return
    else:
        resp = input("[CONFIRM] Proceed? (y/n): ").strip().lower()
        if resp != "y":
            print("Aborted.")
            fc.close()
            return

    # ── Preflight ─────────────────────────────────────────────
    if not args.sitl and not fc.preflight():
        fc.close()
        return

    # ── Load YOLO model ───────────────────────────────────────
    print("[*] Loading YOLO model...")
    model = load_yolo(args.weights, imgsz=args.imgsz)

    # ── Open camera ───────────────────────────────────────────
    cap = open_camera(sitl=args.sitl)
    if not cap and not args.sitl:
        print("[!] No camera — cannot detect X. Aborting.")
        fc.close()
        return

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
        video_path_tmp = f"landing_{ts}_tmp.mp4"
        video_path = f"landing_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(video_path_tmp, fourcc, actual_fps,
                             (FRAME_W, FRAME_H))
        if not vw.isOpened():
            print(f"[!] WARNING: Could not open video writer — recording disabled")
            vw = None
        else:
            print(f"[REC] Recording → {video_path}  ({actual_fps:.1f} FPS)")

    # ── Logging ───────────────────────────────────────────────
    log_fname, log_f = create_log("flyto_land")
    log(log_f, "FLY TO COORDS → DETECT X → LAND ON X (with video)")
    log(log_f, f"Target: ({target_lat:.8f}, {target_lon:.8f})")
    log(log_f, f"From:   ({cur_lat:.8f}, {cur_lon:.8f})")
    log(log_f, f"Distance: {dist:.1f}m  Bearing: {brng:.1f}° ({compass})")
    log(log_f, f"Alt={args.alt}m | Speed={args.speed}m/s | "
               f"Step={DESCEND_STEP}m | Final={FINAL_ALT}m | Land={LAND_ALT}m")
    if video_path:
        log(log_f, f"Video: {video_path}")

    # ── Set cruise speed ──────────────────────────────────────
    set_speed(fc, args.speed)

    with SafeFlight(fc, camera=cap, video_writer=vw) as sf:

        state = "TAKEOFF"
        last_x = 0
        search_t0 = 0
        descend_tgt = 0

        # ══════════════════════════════════════════════════════
        # STATE MACHINE
        # ══════════════════════════════════════════════════════
        while state not in ("DONE", "ABORT"):
            fc.poll()
            cur_alt = fc.alt if fc.alt > 0.3 else args.alt

            # Read camera + detect (used by CV states)
            det = None
            frame = None
            if cap:
                ret, frame = cap.read()
                if ret:
                    det = detect_x(frame, model, args.conf, args.imgsz)

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

                # Stabilize 3 seconds — record frames during settle
                log(log_f, f"At {fc.alt:.1f}m — stabilizing {SETTLE_TIME}s")
                t0 = time.time()
                while time.time() - t0 < SETTLE_TIME:
                    fc.poll()
                    if cap and vw:
                        ret, frm = cap.read()
                        if ret:
                            d = detect_x(frm, model, args.conf, args.imgsz)
                            overlay = draw_overlay(frm.copy(), "TAKEOFF", d,
                                                   fc.alt, fc)
                            vw.write(overlay)
                            frame_count += 1
                    time.sleep(0.05)

                # Skip GPS nav if already at target
                if dist < 1:
                    log(log_f, f"Already within {dist:.2f}m of target — skipping nav")
                    state = "SETTLE_CV"
                else:
                    state = "NAVIGATE"
                continue

            # ── NAVIGATE (fly to GPS coords) ─────────────────
            elif state == "NAVIGATE":
                log(log_f, f"NAVIGATING to ({target_lat:.8f}, {target_lon:.8f})")
                print(f"\n[FLIGHT] Flying to target ({dist:.1f}m {compass}) …\n")

                nav_t0 = time.time()
                last_cmd = 0

                while True:
                    fc.poll()
                    now = time.time()

                    # Resend goto command every 2 seconds
                    if now - last_cmd >= 2.0:
                        send_goto(fc, target_lat, target_lon, args.alt)
                        last_cmd = now

                    # Check distance remaining
                    remaining = haversine(fc.lat, fc.lon, target_lat, target_lon)

                    # Progress bar
                    pct = max(0, (1 - remaining / dist)) * 100 if dist > 0 else 100
                    bar_len = 30
                    filled = int(bar_len * pct / 100)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"\r  {bar} {pct:5.1f}%  "
                          f"rem={remaining:.1f}m  "
                          f"alt={fc.alt:.1f}m  "
                          f"sats={fc.satellites}  "
                          f"batt={fc.battery_pct}%  ",
                          end="", flush=True)

                    # Record during navigation
                    if cap and vw:
                        ret, frm = cap.read()
                        if ret:
                            d = detect_x(frm, model, args.conf, args.imgsz)
                            overlay = draw_overlay(frm.copy(), "NAVIGATE", d,
                                                   fc.alt, fc)
                            vw.write(overlay)
                            frame_count += 1

                    # Arrived?
                    if remaining <= ARRIVE_RADIUS:
                        print()
                        log(log_f, f"ARRIVED — {remaining:.2f}m from target")
                        print(f"\n[FLIGHT] ✓ Arrived at target! ({remaining:.2f}m away)")
                        break

                    # Timeout safety
                    if now - nav_t0 > NAV_TIMEOUT:
                        print()
                        log(log_f, f"NAV TIMEOUT ({NAV_TIMEOUT}s) → RTL")
                        print(f"\n[FLIGHT] ⚠ Navigation timeout — RTL")
                        fc.set_rtl()
                        state = "ABORT"
                        break

                    time.sleep(POLL_RATE)

                if state == "ABORT":
                    continue

                fc.stop()
                state = "SETTLE_CV"
                continue

            # ── SETTLE_CV (hover 3s before starting CV) ──────
            elif state == "SETTLE_CV":
                log(log_f, f"Hovering {SETTLE_TIME}s before CV detection")
                print(f"\n[FLIGHT] Settling {SETTLE_TIME}s before starting detection …")

                t0 = time.time()
                while time.time() - t0 < SETTLE_TIME:
                    fc.poll()
                    elapsed = time.time() - t0
                    print(f"\r  Settling: {elapsed:.1f}s / {SETTLE_TIME:.0f}s  "
                          f"alt={fc.alt:.1f}m  "
                          f"pos=({fc.lat:.8f}, {fc.lon:.8f})  ",
                          end="", flush=True)

                    if cap and vw:
                        ret, frm = cap.read()
                        if ret:
                            d = detect_x(frm, model, args.conf, args.imgsz)
                            overlay = draw_overlay(frm.copy(), "SETTLING", d,
                                                   fc.alt, fc)
                            vw.write(overlay)
                            frame_count += 1
                    time.sleep(0.1)

                print()
                log(log_f, "Settle complete — starting X detection")
                state = "SEARCH"
                search_t0 = time.time()
                continue

            # ── Write video frame (for all CV states) ────────
            if frame is not None and vw:
                overlay = draw_overlay(frame.copy(), state, det, cur_alt, fc,
                                       centered=False)
                vw.write(overlay)
                frame_count += 1
                if record_t0 is None:
                    record_t0 = time.time()

            # ── SEARCH ───────────────────────────────────────
            if state == "SEARCH":
                if search_t0 == 0:
                    search_t0 = time.time()
                elapsed = time.time() - search_t0

                if det:
                    log(log_f, f"X FOUND conf={det['conf']:.2f} "
                               f"@({det['cx']},{det['cy']})")
                    state = "ACQUIRE"
                    last_x = time.time()
                    continue

                if elapsed > SEARCH_TIMEOUT:
                    log(log_f, f"SEARCH TIMEOUT ({SEARCH_TIMEOUT}s) → RTL")
                    fc.set_rtl()
                    state = "ABORT"
                    continue

                print(f"\r  [SEARCH] {elapsed:.0f}s / {SEARCH_TIMEOUT:.0f}s | "
                      f"Alt={cur_alt:.1f}m   ", end="", flush=True)
                time.sleep(0.05)

            # ── ACQUIRE (center on X) ────────────────────────
            elif state == "ACQUIRE":
                if det is None:
                    lost = time.time() - last_x
                    if lost > LOST_TIMEOUT:
                        log(log_f, f"LOST X {lost:.0f}s → RTL")
                        fc.stop()
                        fc.set_rtl()
                        state = "ABORT"
                        continue
                    fc.stop()
                    print(f"\r  [ACQUIRE] Lost X — holding ({lost:.1f}s / "
                          f"{LOST_TIMEOUT:.0f}s)   ", end="", flush=True)
                    time.sleep(VEL_RATE)
                    continue

                last_x = time.time()
                dx_px = det['cx'] - FRAME_W // 2
                dy_px = det['cy'] - FRAME_H // 2

                # Adaptive deadzone + speed based on altitude
                dz = DEADZONE_LOW if cur_alt < FINAL_ALT + 1 else DEADZONE_HIGH
                spd = SPEED_LOW if cur_alt < FINAL_ALT + 1 else SPEED_HIGH

                if abs(dx_px) <= dz and abs(dy_px) <= dz:
                    # CENTERED
                    log(log_f, f"CENTERED at {cur_alt:.1f}m "
                               f"(offset: {dx_px:+d},{dy_px:+d}px)")
                    fc.stop()

                    # Re-draw this frame with centered flag for the video
                    if frame is not None and vw:
                        overlay = draw_overlay(frame.copy(), state, det,
                                               cur_alt, fc, centered=True)
                        vw.write(overlay)
                        frame_count += 1

                    if cur_alt <= LAND_ALT + 0.5:
                        state = "LAND"
                    elif cur_alt <= FINAL_ALT + 0.5:
                        state = "FINAL"
                    else:
                        state = "DESCEND"
                        descend_tgt = max(cur_alt - DESCEND_STEP, FINAL_ALT)
                    time.sleep(0.5)
                    continue

                # Compute correction
                m_fwd, m_right = pixels_to_meters(dx_px, dy_px, cur_alt)
                d = math.sqrt(m_fwd**2 + m_right**2)
                scale = min(spd / d, 1.0) if d > spd else 0.5
                vx = m_fwd * scale
                vy = m_right * scale

                fc.velocity_body(vx, vy, 0)

                parts = []
                if abs(m_fwd) > 0.05:
                    parts.append(f"{'FWD' if m_fwd > 0 else 'BACK'} {abs(m_fwd):.2f}m")
                if abs(m_right) > 0.05:
                    parts.append(f"{'RIGHT' if m_right > 0 else 'LEFT'} {abs(m_right):.2f}m")
                print(f"\r  [ACQUIRE] {' + '.join(parts) or '~'} | "
                      f"v=({vx:.2f},{vy:.2f}) | Alt={cur_alt:.1f}m | "
                      f"conf={det['conf']:.2f}   ", end="", flush=True)
                time.sleep(VEL_RATE)

            # ── DESCEND ──────────────────────────────────────
            elif state == "DESCEND":
                log(log_f, f"DESCEND {cur_alt:.1f}m → {descend_tgt:.1f}m")
                t0 = time.time()
                while True:
                    fc.poll()
                    cur_alt = fc.alt
                    if cur_alt <= descend_tgt + 0.3:
                        break
                    if time.time() - t0 > 15:
                        break
                    fc.velocity_ned(0, 0, DESCENT_VZ)

                    # Keep recording during descent
                    if cap:
                        ret, frm = cap.read()
                        if ret:
                            d = detect_x(frm, model, args.conf, args.imgsz)
                            overlay = draw_overlay(frm.copy(), "DESCEND", d,
                                                   cur_alt, fc)
                            if vw:
                                vw.write(overlay)
                                frame_count += 1

                    print(f"\r  [DESCEND] {cur_alt:.1f}m → {descend_tgt:.1f}m   ",
                          end="", flush=True)
                    time.sleep(VEL_RATE)

                fc.stop()
                log(log_f, f"At {cur_alt:.1f}m — re-acquiring")
                time.sleep(1)
                state = "ACQUIRE"

            # ── FINAL APPROACH ───────────────────────────────
            elif state == "FINAL":
                log(log_f, f"FINAL APPROACH at {cur_alt:.1f}m")
                t0 = time.time()
                while True:
                    fc.poll()
                    cur_alt = fc.alt

                    if cap:
                        ret, frame = cap.read()
                        if ret:
                            det = detect_x(frame, model, args.conf, args.imgsz)

                    # Record during final approach
                    if frame is not None and vw:
                        overlay = draw_overlay(frame.copy(), "FINAL", det,
                                               cur_alt, fc)
                        vw.write(overlay)
                        frame_count += 1

                    if cur_alt <= LAND_ALT + 0.3:
                        log(log_f, f"Below {LAND_ALT}m → LAND")
                        state = "LAND"
                        break
                    if time.time() - t0 > 30:
                        log(log_f, "Final timeout → LAND")
                        state = "LAND"
                        break

                    if det:
                        last_x = time.time()
                        dx_px = det['cx'] - FRAME_W // 2
                        dy_px = det['cy'] - FRAME_H // 2
                        m_fwd, m_right = pixels_to_meters(dx_px, dy_px, cur_alt)
                        d = math.sqrt(m_fwd**2 + m_right**2)
                        sc = min(SPEED_LOW / d, 1.0) if d > SPEED_LOW else 0.4
                        vx, vy = m_fwd * sc, m_right * sc
                        fc.velocity_body(vx, vy, 0.15)
                        print(f"\r  [FINAL] Alt={cur_alt:.1f}m | "
                              f"({dx_px:+d},{dy_px:+d})px | "
                              f"v=({vx:.2f},{vy:.2f},0.15)   ",
                              end="", flush=True)
                    else:
                        lost = time.time() - last_x
                        fc.stop()
                        print(f"\r  [FINAL] Lost X ({lost:.1f}s) | "
                              f"Alt={cur_alt:.1f}m   ", end="", flush=True)
                        if lost > 5:
                            log(log_f, "Lost in final 5s → LAND anyway")
                            state = "LAND"
                            break
                    time.sleep(VEL_RATE)

            # ── LAND ─────────────────────────────────────────
            elif state == "LAND":
                log(log_f, f"LAND at {cur_alt:.1f}m")
                fc.set_land()

                # Keep recording while landing
                land_t0 = time.time()
                while fc.armed and (time.time() - land_t0 < 30):
                    fc.poll()
                    if cap:
                        ret, frm = cap.read()
                        if ret:
                            d = detect_x(frm, model, args.conf, args.imgsz)
                            overlay = draw_overlay(frm.copy(), "LANDING",
                                                   d, fc.alt, fc)
                            if vw:
                                vw.write(overlay)
                                frame_count += 1
                    time.sleep(0.1)

                final_dist = haversine(fc.lat, fc.lon, target_lat, target_lon)
                log(log_f, "LANDED ON X!")
                log(log_f, f"Final position: ({fc.lat:.8f}, {fc.lon:.8f})")
                log(log_f, f"Distance from target coords: {final_dist:.2f}m")

                print("\n\n" + "=" * 60)
                print("  ★ ★ ★  LANDED ON X!  ★ ★ ★")
                print(f"  Final position: ({fc.lat:.8f}, {fc.lon:.8f})")
                print(f"  Distance from GPS target: {final_dist:.2f}m")
                print("=" * 60 + "\n")
                state = "DONE"

            # ── ABORT ────────────────────────────────────────
            elif state == "ABORT":
                log(log_f, "ABORTED")
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


def build_parser():
    p = argparse.ArgumentParser(
        description="Fly to GPS coordinates, detect X, and land on it")
    p.add_argument("--lat", type=float, required=True,
                   help="Target latitude (decimal degrees)")
    p.add_argument("--lon", type=float, required=True,
                   help="Target longitude (decimal degrees)")
    p.add_argument("--alt", type=float, default=TAKEOFF_ALT,
                   help=f"Flight altitude in meters (default {TAKEOFF_ALT})")
    p.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                   help=f"Cruise speed in m/s (default {DEFAULT_SPEED})")
    p.add_argument("--weights", default="best_22.pt",
                   help="YOLO weights file (default: best_22.pt)")
    p.add_argument("--conf", type=float, default=0.50,
                   help="YOLO confidence threshold (default: 0.50)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="YOLO input size (default: 640)")
    p.add_argument("--sitl", action="store_true",
                   help="SITL mode (uses webcam)")
    return p


if __name__ == "__main__":
    main()
