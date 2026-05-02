#!/usr/bin/env python3
"""
Script 6: Fly Forward → Detect X → Land on X
================================================
Takeoff to ALT → move DISTANCE forward → hover & search for X →
center over X → descend in steps → land on X.

Combines the forward-move logic from 4_fly_and_move.py with the
full X-landing state machine from 5_land_on_x.py.

Usage:
    Terminal 1: mavproxy.py --master=/dev/ttyACM0 --baudrate=115200 --out=udp:127.0.0.1:14551
    Terminal 2: python3 6_fly_forward_land_on_x.py --alt 5 --distance 10

    SITL:
    Terminal 1: sim_vehicle.py -v ArduCopter --console --map --out=udp:127.0.0.1:14551
    Terminal 2: python3 6_fly_forward_land_on_x.py --alt 5 --distance 10 --sitl

Failsafes:
    Ctrl+C → RTL | Exception → RTL | X lost 10s → RTL | Search timeout 60s → RTL
"""
import argparse, time, math, os, cv2
from datetime import datetime
from flight_utils import (FlightController, SafeFlight, open_camera,
                          load_yolo, detect_x, pixels_to_meters,
                          get_camera_fps,
                          TAKEOFF_ALT, FRAME_W, FRAME_H,
                          confirm, create_log, log)

# ── Forward move parameters ───────────────────────────────────
MOVE_SPEED = 0.5     # m/s forward speed
SETTLE = 3.0         # seconds to settle after move

# ── X-landing parameters ─────────────────────────────────────
DESCEND_STEP = 1.0       # meters per step
FINAL_ALT = 1.5          # switch to final approach below this
LAND_ALT = 0.8           # trigger LAND below this

DEADZONE_HIGH = 60       # px — centered at high altitude
DEADZONE_LOW = 30        # px — centered near ground
SPEED_HIGH = 0.3         # m/s — centering speed at altitude
SPEED_LOW = 0.15         # m/s — centering speed near ground

LOST_TIMEOUT = 10.0      # seconds lost → RTL
SEARCH_TIMEOUT = 60.0    # seconds searching → RTL
DESCENT_VZ = 0.3         # m/s descent rate
VEL_RATE = 0.2           # seconds between velocity commands

# ── Video / overlay ───────────────────────────────────────────
OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
OVERLAY_COLOR_OK = (0, 255, 0)
OVERLAY_COLOR_LOST = (0, 0, 255)
OVERLAY_COLOR_CENTER = (0, 255, 255)


def draw_overlay(frame, state, det, cur_alt, fc, centered=False):
    """Draw detection overlay, crosshair, and HUD onto the frame."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Crosshair
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

    # HUD
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


def move_forward(fc, log_f, distance, cap=None, vw=None, model=None, args=None):
    """
    Move forward by 'distance' meters using body velocity.
    Records video frames during the move if camera is available.
    """
    move_time = distance / MOVE_SPEED
    log(log_f, f"Moving FORWARD {distance}m at {MOVE_SPEED}m/s ({move_time:.1f}s)")

    frame_count = 0
    t0 = time.time()
    while time.time() - t0 < move_time:
        fc.poll()
        fc.velocity_body(MOVE_SPEED, 0, 0)  # forward, no lateral, no vertical

        elapsed = time.time() - t0
        print(f"\r  [MOVE] FORWARD... {elapsed:.1f}s / {move_time:.1f}s | "
              f"Alt={fc.alt:.1f}m   ", end="", flush=True)

        # Record video during move
        if cap and vw:
            ret, frame = cap.read()
            if ret:
                det = detect_x(frame, model, args.conf, args.imgsz) if model else None
                overlay = draw_overlay(frame.copy(), "MOVING FWD", det, fc.alt, fc)
                vw.write(overlay)
                frame_count += 1

        time.sleep(0.2)

    # Stop and settle
    fc.stop()
    print(f"\n[*] Forward move done. Settling {SETTLE}s...")
    log(log_f, f"Forward move complete. Settling {SETTLE}s")

    settle_t0 = time.time()
    while time.time() - settle_t0 < SETTLE:
        fc.poll()
        if cap and vw:
            ret, frame = cap.read()
            if ret:
                det = detect_x(frame, model, args.conf, args.imgsz) if model else None
                overlay = draw_overlay(frame.copy(), "SETTLING", det, fc.alt, fc)
                vw.write(overlay)
                frame_count += 1
        time.sleep(0.1)

    return frame_count


def main(args):
    if not args.dry_run and not args.sitl:
        if not confirm("6_fly_forward_land_on_x.py",
                       f"Takeoff {args.alt}m → Move {args.distance}m FORWARD → "
                       f"Detect X → Center → Descend → LAND ON X\n"
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
        print("[!] No camera — cannot detect X."); fc.close(); return

    # ── Video writer setup ────────────────────────────────────
    vw = None
    video_path = None
    video_path_tmp = None
    actual_fps = 20.0
    frame_count = 0
    record_t0 = None

    if cap:
        actual_fps = get_camera_fps(cap)
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

    log_fname, log_f = create_log("fly_forward_land")
    log(log_f, "FLY FORWARD → DETECT X → LAND ON X (with video)")
    log(log_f, f"Alt={args.alt}m | Distance={args.distance}m | "
               f"Step={DESCEND_STEP}m | Final={FINAL_ALT}m | Land={LAND_ALT}m")
    if video_path:
        log(log_f, f"Video: {video_path}")

    with SafeFlight(fc, camera=cap, video_writer=vw) as sf:

        state = "TAKEOFF"
        last_x = 0
        search_t0 = 0
        descend_tgt = 0

        if args.dry_run:
            state = "SEARCH"
            log(log_f, "DRY RUN — skipping takeoff and move")

        # ══════════════════════════════════════════════════════
        # STATE MACHINE
        # ══════════════════════════════════════════════════════
        while state not in ("DONE", "ABORT"):
            if not args.dry_run:
                fc.poll()
            cur_alt = fc.alt if (not args.dry_run and fc.alt > 0.3) else args.alt

            # Read camera + detect
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

                log(log_f, "At altitude — stabilizing 3s")
                t0 = time.time()
                while time.time() - t0 < 3:
                    if cap:
                        ret, frm = cap.read()
                        if ret:
                            d = detect_x(frm, model, args.conf, args.imgsz)
                            fc.poll()
                            overlay = draw_overlay(frm.copy(), "TAKEOFF", d,
                                                   fc.alt, fc)
                            if vw:
                                vw.write(overlay); frame_count += 1
                    time.sleep(0.05)

                state = "MOVE_FORWARD"
                continue

            # ── MOVE FORWARD ─────────────────────────────────
            elif state == "MOVE_FORWARD":
                log(log_f, f"MOVE FORWARD {args.distance}m")
                fwd_frames = move_forward(fc, log_f, args.distance,
                                          cap=cap, vw=vw, model=model,
                                          args=args)
                frame_count += fwd_frames
                log(log_f, "Forward move complete — starting X detection")
                state = "SEARCH"
                search_t0 = time.time()
                continue

            # ── Write video frame (for all states after move) ──
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
                    if not args.dry_run:
                        fc.set_rtl()
                    state = "ABORT"; continue

                print(f"\r  [SEARCH] {elapsed:.0f}s / {SEARCH_TIMEOUT:.0f}s | "
                      f"Alt={cur_alt:.1f}m   ", end="", flush=True)
                time.sleep(0.05)

            # ── ACQUIRE (center on X) ────────────────────────
            elif state == "ACQUIRE":
                if det is None:
                    lost = time.time() - last_x
                    if lost > LOST_TIMEOUT:
                        log(log_f, f"LOST X {lost:.0f}s → RTL")
                        if not args.dry_run:
                            fc.stop(); fc.set_rtl()
                        state = "ABORT"; continue
                    if not args.dry_run:
                        fc.stop()
                    print(f"\r  [ACQUIRE] Lost X — holding ({lost:.1f}s / "
                          f"{LOST_TIMEOUT:.0f}s)   ", end="", flush=True)
                    time.sleep(VEL_RATE)
                    continue

                last_x = time.time()
                dx_px = det['cx'] - FRAME_W // 2
                dy_px = det['cy'] - FRAME_H // 2

                # Adaptive deadzone + speed
                dz = DEADZONE_LOW if cur_alt < FINAL_ALT + 1 else DEADZONE_HIGH
                spd = SPEED_LOW if cur_alt < FINAL_ALT + 1 else SPEED_HIGH

                if abs(dx_px) <= dz and abs(dy_px) <= dz:
                    # CENTERED
                    log(log_f, f"CENTERED at {cur_alt:.1f}m "
                               f"(offset: {dx_px:+d},{dy_px:+d}px)")
                    if not args.dry_run:
                        fc.stop()

                    # Re-draw frame with centered flag
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
                dist = math.sqrt(m_fwd**2 + m_right**2)
                scale = min(spd / dist, 1.0) if dist > spd else 0.5
                vx = m_fwd * scale
                vy = m_right * scale

                if not args.dry_run:
                    fc.velocity_body(vx, vy, 0)

                parts = []
                if abs(m_fwd) > 0.05:
                    parts.append(f"{'FWD' if m_fwd>0 else 'BACK'} {abs(m_fwd):.2f}m")
                if abs(m_right) > 0.05:
                    parts.append(f"{'RIGHT' if m_right>0 else 'LEFT'} {abs(m_right):.2f}m")
                print(f"\r  [ACQUIRE] {' + '.join(parts) or '~'} | "
                      f"v=({vx:.2f},{vy:.2f}) | Alt={cur_alt:.1f}m | "
                      f"conf={det['conf']:.2f}   ", end="", flush=True)
                time.sleep(VEL_RATE)

            # ── DESCEND ──────────────────────────────────────
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

                    # Keep recording during descent
                    if cap:
                        ret, frm = cap.read()
                        if ret:
                            d = detect_x(frm, model, args.conf, args.imgsz)
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

            # ── FINAL APPROACH ───────────────────────────────
            elif state == "FINAL":
                log(log_f, f"FINAL APPROACH at {cur_alt:.1f}m")
                t0 = time.time()
                while True:
                    if not args.dry_run:
                        fc.poll()
                    cur_alt = fc.alt if not args.dry_run else LAND_ALT

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
                        state = "LAND"; break
                    if time.time() - t0 > 30:
                        log(log_f, "Final timeout → LAND")
                        state = "LAND"; break

                    if det:
                        last_x = time.time()
                        dx_px = det['cx'] - FRAME_W // 2
                        dy_px = det['cy'] - FRAME_H // 2
                        m_fwd, m_right = pixels_to_meters(dx_px, dy_px, cur_alt)
                        dist = math.sqrt(m_fwd**2 + m_right**2)
                        sc = min(SPEED_LOW / dist, 1.0) if dist > SPEED_LOW else 0.4
                        vx, vy = m_fwd * sc, m_right * sc
                        if not args.dry_run:
                            fc.velocity_body(vx, vy, 0.15)
                        print(f"\r  [FINAL] Alt={cur_alt:.1f}m | "
                              f"({dx_px:+d},{dy_px:+d})px | "
                              f"v=({vx:.2f},{vy:.2f},0.15)   ",
                              end="", flush=True)
                    else:
                        lost = time.time() - last_x
                        if not args.dry_run:
                            fc.stop()
                        print(f"\r  [FINAL] Lost X ({lost:.1f}s) | "
                              f"Alt={cur_alt:.1f}m   ", end="", flush=True)
                        if lost > 5:
                            log(log_f, "Lost in final 5s → LAND anyway")
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
                                d = detect_x(frm, model, args.conf, args.imgsz)
                                overlay = draw_overlay(frm.copy(), "LANDING",
                                                       d, fc.alt, fc)
                                if vw:
                                    vw.write(overlay); frame_count += 1
                        time.sleep(0.1)
                log(log_f, "LANDED ON X!")
                print("\n\n" + "=" * 55)
                print("  ★ ★ ★  LANDED ON X!  ★ ★ ★")
                print("=" * 55 + "\n")
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
        description="Fly forward then detect X and land on it")
    p.add_argument("--alt", type=float, default=TAKEOFF_ALT,
                   help="Takeoff altitude in meters (default: 5)")
    p.add_argument("--distance", type=float, default=10.0,
                   help="Forward distance in meters (default: 10)")
    p.add_argument("--weights", default="best_22.pt")
    p.add_argument("--conf", type=float, default=0.50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--dry-run", action="store_true",
                   help="Camera-only, no flight (tests detection)")
    p.add_argument("--sitl", action="store_true",
                   help="SITL mode (uses webcam)")
    main(p.parse_args())
