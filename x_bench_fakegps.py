#!/usr/bin/env python3
"""
Minimal Fake GPS Test
======================
Just sends fake GPS and prints EVERYTHING Pixhawk says back.
No camera, no YOLO, no complicated logic.
Use this to debug whether fake GPS is working.

PREREQUISITE:
  1. GPS_TYPE = 14 in Mission Planner
  2. ARMING_CHECK = 0
  3. Unplug real GPS hardware from Pixhawk!
  4. Reboot Pixhawk after all changes
  5. Connect via USB-C

Usage:
    python3 test_fakegps.py
    python3 test_fakegps.py --device /dev/ttyACM1
"""

import argparse
import time
from datetime import datetime
from pymavlink import mavutil

GPS_EPOCH = datetime(1980, 1, 6)


def get_gps_time():
    now = datetime.utcnow()
    delta = now - GPS_EPOCH
    gps_week = delta.days // 7
    gps_week_ms = ((delta.days % 7) * 86400 + delta.seconds) * 1000
    return gps_week, gps_week_ms


def main(args):
    # Connect
    print(f"[*] Connecting to {args.device} at {args.baud}...")
    master = mavutil.mavlink_connection(args.device, baud=args.baud)
    print("[*] Waiting for heartbeat...")
    master.wait_heartbeat(timeout=30)
    print(f"[*] Connected! system={master.target_system}")

    # Check current GPS_TYPE
    print("\n[*] Requesting GPS_TYPE parameter...")
    master.mav.param_request_read_send(
        master.target_system, master.target_component,
        b'GPS_TYPE', -1
    )
    msg = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=5)
    if msg:
        print(f"[*] GPS_TYPE = {int(msg.param_value)}")
        if int(msg.param_value) != 14:
            print("[!] WARNING: GPS_TYPE is NOT 14!")
            print("    Set GPS_TYPE=14 in Mission Planner, then REBOOT Pixhawk")
            print("    Also UNPLUG the real GPS module!")
            return
    else:
        print("[!] Could not read GPS_TYPE")

    # Also check ARMING_CHECK
    master.mav.param_request_read_send(
        master.target_system, master.target_component,
        b'ARMING_CHECK', -1
    )
    msg = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=5)
    if msg:
        print(f"[*] ARMING_CHECK = {int(msg.param_value)}")

    # Gujarat coordinates
    lat = int(23.0258 * 1e7)
    lon = int(72.5873 * 1e7)
    alt = 10.0

    print(f"\n[*] Sending fake GPS: lat=23.0258, lon=72.5873, alt={alt}m")
    print(f"[*] Sending at 10Hz + heartbeat at 1Hz")
    print(f"[*] Watching for ALL messages from Pixhawk...")
    print(f"[*] Keep Pixhawk STILL. Press Ctrl+C to stop.")
    print("=" * 70)

    last_heartbeat = 0
    last_gps_send = 0
    start_time = time.time()
    got_fix = False

    try:
        while True:
            now = time.time()
            elapsed = now - start_time

            # Send companion heartbeat at 1Hz
            if now - last_heartbeat >= 1.0:
                master.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0
                )
                last_heartbeat = now

            # Send GPS at 10Hz
            if now - last_gps_send >= 0.1:
                gps_week, gps_week_ms = get_gps_time()
                master.mav.gps_input_send(
                    int(now * 1e6),     # time_usec
                    0,                   # gps_id
                    8,                   # ignore_flags (only ignore speed_accuracy)
                    gps_week_ms,        # time_week_ms
                    gps_week,           # time_week
                    3,                   # fix_type: 3D
                    lat,                 # lat
                    lon,                 # lon
                    alt,                 # alt
                    0.6,                 # hdop
                    0.6,                 # vdop
                    0.0, 0.0, 0.0,      # vn, ve, vd
                    0.0,                 # speed_accuracy
                    0.3,                 # horiz_accuracy
                    0.5,                 # vert_accuracy
                    16,                  # satellites_visible
                )
                last_gps_send = now

            # Read ALL messages from Pixhawk
            msg = master.recv_match(blocking=False)
            if msg:
                mtype = msg.get_type()

                # Print interesting messages
                if mtype == 'GPS_RAW_INT':
                    fix = msg.fix_type
                    sats = msg.satellites_visible
                    lat_r = msg.lat / 1e7
                    lon_r = msg.lon / 1e7
                    alt_r = msg.alt / 1000.0
                    marker = " <<<< 3D FIX!" if fix >= 3 else ""
                    print(f"[{elapsed:6.1f}s] GPS_RAW_INT: fix={fix} sats={sats} "
                          f"lat={lat_r:.4f} lon={lon_r:.4f} alt={alt_r:.1f}m{marker}")
                    if fix >= 3 and not got_fix:
                        got_fix = True
                        print(f"\n{'='*70}")
                        print(f"  *** GOT 3D FIX after {elapsed:.0f}s! ***")
                        print(f"{'='*70}\n")

                elif mtype == 'GPS2_RAW':
                    fix = msg.fix_type
                    sats = msg.satellites_visible
                    print(f"[{elapsed:6.1f}s] GPS2_RAW: fix={fix} sats={sats}")

                elif mtype == 'EKF_STATUS_REPORT':
                    f = msg.flags
                    print(f"[{elapsed:6.1f}s] EKF: flags=0x{f:04x} "
                          f"att={'Y' if f&1 else 'N'} "
                          f"vel_h={'Y' if f&2 else 'N'} "
                          f"vel_v={'Y' if f&4 else 'N'} "
                          f"pos_rel={'Y' if f&8 else 'N'} "
                          f"pos_abs={'Y' if f&16 else 'N'} "
                          f"pos_vert={'Y' if f&32 else 'N'}")

                elif mtype == 'STATUSTEXT':
                    text = msg.text.strip('\x00').strip()
                    if text:
                        print(f"[{elapsed:6.1f}s] *** PIXHAWK: {text} ***")

                elif mtype == 'HEARTBEAT' and msg.get_srcSystem() == master.target_system:
                    mode = msg.custom_mode
                    armed = bool(msg.base_mode & 128)
                    print(f"[{elapsed:6.1f}s] HEARTBEAT: mode={mode} armed={armed}")

                elif mtype == 'SYS_STATUS':
                    # Print sensor health occasionally
                    pass  # too noisy

            else:
                time.sleep(0.01)  # small sleep when no messages

    except KeyboardInterrupt:
        print(f"\n\n[*] Stopped after {time.time() - start_time:.0f}s")
        if got_fix:
            print("[*] 3D fix WAS obtained — fake GPS is working!")
            print("[*] You can now run x_bench_fakegps.py")
        else:
            print("[!] Never got 3D fix.")
            print("    Checklist:")
            print("    1. Is GPS_TYPE = 14? (check in Mission Planner)")
            print("    2. Did you REBOOT Pixhawk after setting it?")
            print("    3. Did you UNPLUG the real GPS module?")
            print("    4. Was Pixhawk kept still?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Fake GPS Test")
    parser.add_argument("--device", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()
    main(args)
