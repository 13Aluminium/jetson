#!/usr/bin/env python3

"""
pixhawk_port_inspector.py

Purpose:
- Connect to Pixhawk over MAVLink
- Read SERIALx parameters
- Infer likely device roles on Pixhawk serial ports
- Listen for MAVLink messages that suggest GPS, lidar, telemetry, etc.

Usage:
    python3 pixhawk_port_inspector.py --port /dev/ttyACM0 --baud 115200
"""

import argparse
import time
from collections import defaultdict
from pymavlink import mavutil

SERIAL_PARAMS = [
    "SERIAL1_PROTOCOL", "SERIAL1_BAUD",
    "SERIAL2_PROTOCOL", "SERIAL2_BAUD",
    "SERIAL3_PROTOCOL", "SERIAL3_BAUD",
    "SERIAL4_PROTOCOL", "SERIAL4_BAUD",
    "SERIAL5_PROTOCOL", "SERIAL5_BAUD",
    "SERIAL6_PROTOCOL", "SERIAL6_BAUD",
]

PROTO_HINTS = {
    0: "Disabled",
    1: "MAVLink 1",
    2: "MAVLink 2",
    5: "GPS",
    9: "Rangefinder / Lidar / Distance sensor",
    10: "FrSky / telemetry-related",
}

MESSAGE_HINTS = {
    "DISTANCE_SENSOR": "Lidar / rangefinder data is present",
    "RANGEFINDER": "Rangefinder data is present",
    "GPS_RAW_INT": "GPS data is present",
    "GLOBAL_POSITION_INT": "Navigation solution is present",
    "RADIO_STATUS": "Telemetry radio status is present",
    "HEARTBEAT": "MAVLink system/component is active",
}

def request_param(master, name, timeout=2.0):
    master.mav.param_request_read_send(
        master.target_system,
        master.target_component,
        name.encode("utf-8"),
        -1
    )

    start = time.time()
    while time.time() - start < timeout:
        msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if msg and msg.param_id.decode("utf-8").rstrip("\x00") == name:
            return msg.param_value
    return None

def protocol_name(value):
    if value is None:
        return "Unknown"
    iv = int(value)
    return PROTO_HINTS.get(iv, f"Protocol ID {iv}")

def likely_port_label(serial_num):
    mapping = {
        1: "Usually TELEM1 on many setups",
        2: "Usually TELEM2 on many setups",
        3: "Often GPS or another serial/UART",
    }
    return mapping.get(serial_num, "Board-dependent serial port")

def inspect_params(master):
    print("\n=== Pixhawk SERIAL Parameter Inspection ===\n")
    results = {}

    for i in range(1, 7):
        proto_name = f"SERIAL{i}_PROTOCOL"
        baud_name = f"SERIAL{i}_BAUD"

        proto_val = request_param(master, proto_name)
        baud_val = request_param(master, baud_name)

        results[i] = {
            "protocol_raw": proto_val,
            "baud_raw": baud_val,
        }

        proto_desc = protocol_name(proto_val)
        baud_desc = int(baud_val) if baud_val is not None else "Unknown"

        print(f"Port SERIAL{i} ({likely_port_label(i)})")
        print(f"  {proto_name}: {proto_val} -> {proto_desc}")
        print(f"  {baud_name}: {baud_desc}")

        guess = infer_from_protocol(proto_val)
        print(f"  Likely use: {guess}\n")

    return results

def infer_from_protocol(proto_val):
    if proto_val is None:
        return "Could not determine"
    iv = int(proto_val)

    if iv in (1, 2):
        return "Telemetry / companion computer / radio MAVLink link"
    if iv == 5:
        return "GPS receiver"
    if iv == 9:
        return "Lidar / rangefinder / distance sensor"
    if iv == 0:
        return "Disabled / unused"
    if iv == 10:
        return "Telemetry-related peripheral"
    return f"Unknown or custom use (protocol {iv})"

def sniff_messages(master, duration=10):
    print("\n=== Listening for MAVLink traffic ===\n")
    print(f"Listening for {duration} seconds...\n")

    seen_counts = defaultdict(int)
    first_seen = {}

    start = time.time()
    while time.time() - start < duration:
        msg = master.recv_match(blocking=True, timeout=0.5)
        if not msg:
            continue

        mtype = msg.get_type()
        seen_counts[mtype] += 1
        if mtype not in first_seen:
            first_seen[mtype] = msg

    if not seen_counts:
        print("No MAVLink messages received during sniff window.")
        return seen_counts

    interesting = [
        "HEARTBEAT",
        "RADIO_STATUS",
        "DISTANCE_SENSOR",
        "RANGEFINDER",
        "GPS_RAW_INT",
        "GLOBAL_POSITION_INT",
    ]

    for mtype in interesting:
        if seen_counts[mtype] > 0:
            print(f"{mtype}: {seen_counts[mtype]} messages")
            if mtype in MESSAGE_HINTS:
                print(f"  Hint: {MESSAGE_HINTS[mtype]}")

    print("\nTop message counts:")
    for mtype, count in sorted(seen_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {mtype}: {count}")

    return seen_counts

def summarize(serial_results, seen_counts):
    print("\n=== Summary / Best Guess ===\n")

    for i, vals in serial_results.items():
        proto_val = vals["protocol_raw"]
        guess = infer_from_protocol(proto_val)
        print(f"SERIAL{i}: {guess}")

    if seen_counts.get("RADIO_STATUS", 0) > 0:
        print("\nTelemetry radio appears to be active because RADIO_STATUS messages were seen.")

    if seen_counts.get("DISTANCE_SENSOR", 0) > 0 or seen_counts.get("RANGEFINDER", 0) > 0:
        print("A lidar/rangefinder appears to be active because distance sensor messages were seen.")

    if seen_counts.get("GPS_RAW_INT", 0) > 0:
        print("A GPS appears to be active because GPS messages were seen.")

    print("\nNote:")
    print("This tells you the likely role of each Pixhawk serial port.")
    print("It does NOT guarantee the exact physical device model plugged into TELEM1/TELEM2.")
    print("For exact confirmation, also compare with your wiring and autopilot parameter setup.")

def main():
    parser = argparse.ArgumentParser(description="Inspect Pixhawk serial ports and infer connected devices")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Pixhawk connection port")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--listen", type=int, default=10, help="Seconds to sniff MAVLink messages")
    args = parser.parse_args()

    print("=" * 60)
    print("PIXHAWK PORT INSPECTOR")
    print("=" * 60)
    print(f"Connecting to {args.port} @ {args.baud}...")

    master = mavutil.mavlink_connection(args.port, baud=args.baud)
    master.wait_heartbeat(timeout=10)
    print("Heartbeat received.")
    print(f"System ID: {master.target_system}")
    print(f"Component ID: {master.target_component}")

    serial_results = inspect_params(master)
    seen_counts = sniff_messages(master, duration=args.listen)
    summarize(serial_results, seen_counts)

if __name__ == "__main__":
    main()
