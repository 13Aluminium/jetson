#!/usr/bin/env python3

"""
pixhawk_port_inspector.py

Purpose:
- Connect to Pixhawk over MAVLink
- Read SERIALx parameters
- Infer likely device roles on Pixhawk ports
- Listen for MAVLink messages and guess what devices are active

Examples:
    python3 pixhawk_port_inspector.py
    python3 pixhawk_port_inspector.py --port /dev/ttyACM0 --baud 115200
    python3 pixhawk_port_inspector.py --listen 15

What it can tell you:
- TELEM1 is likely MAVLink radio / telemetry
- TELEM2 is likely lidar / rangefinder
- A GPS is active
- A distance sensor is active

What it cannot guarantee:
- Exact hardware model name from terminal alone
"""

import argparse
import time
from collections import defaultdict
from pymavlink import mavutil


PROTO_HINTS = {
    0: "Disabled",
    1: "MAVLink 1",
    2: "MAVLink 2",
    5: "GPS",
    9: "Rangefinder / Lidar",
    10: "Telemetry-related peripheral",
}


MESSAGE_HINTS = {
    "DISTANCE_SENSOR": "Lidar / rangefinder data is present",
    "RANGEFINDER": "Rangefinder data is present",
    "GPS_RAW_INT": "GPS data is present",
    "GLOBAL_POSITION_INT": "Navigation solution is present",
    "RADIO_STATUS": "Telemetry radio status is present",
    "HEARTBEAT": "A MAVLink system/component is active",
}


def request_param(master, name, timeout=2.0):
    """
    Request a single parameter safely.
    Works whether msg.param_id is bytes or string.
    """
    master.mav.param_request_read_send(
        master.target_system,
        master.target_component,
        name.encode("utf-8"),
        -1
    )

    start = time.time()
    while time.time() - start < timeout:
        msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if not msg:
            continue

        pid = msg.param_id
        if isinstance(pid, bytes):
            pid = pid.decode("utf-8", errors="ignore")

        if pid.rstrip("\x00") == name:
            return msg.param_value

    return None


def protocol_name(value):
    if value is None:
        return "Unknown"
    iv = int(value)
    return PROTO_HINTS.get(iv, f"Protocol ID {iv}")


def port_label(serial_num):
    """
    Board mapping varies, but on many Pixhawk/ArduPilot setups:
    SERIAL1 -> TELEM1
    SERIAL2 -> TELEM2
    """
    mapping = {
        1: "likely TELEM1",
        2: "likely TELEM2",
        3: "likely GPS / UART3",
        4: "serial port 4",
        5: "serial port 5",
        6: "serial port 6",
    }
    return mapping.get(serial_num, f"serial port {serial_num}")


def infer_from_protocol(proto_val):
    if proto_val is None:
        return "Could not determine"

    iv = int(proto_val)

    if iv in (1, 2):
        return "Telemetry / MAVLink device / radio / companion link"
    if iv == 5:
        return "GPS receiver"
    if iv == 9:
        return "Lidar / rangefinder / distance sensor"
    if iv == 10:
        return "Telemetry-related peripheral"
    if iv == 0:
        return "Disabled / unused"

    return f"Unknown or custom use (protocol {iv})"


def inspect_params(master, max_serial=6):
    print("\n=== SERIAL PARAMETER INSPECTION ===\n")
    results = {}

    for i in range(1, max_serial + 1):
        proto_name = f"SERIAL{i}_PROTOCOL"
        baud_name = f"SERIAL{i}_BAUD"

        proto_val = request_param(master, proto_name)
        baud_val = request_param(master, baud_name)

        results[i] = {
            "protocol_raw": proto_val,
            "baud_raw": baud_val,
            "protocol_text": protocol_name(proto_val),
            "guess": infer_from_protocol(proto_val),
        }

        print(f"{proto_name} ({port_label(i)})")
        print(f"  Protocol value : {proto_val}")
        print(f"  Protocol name  : {protocol_name(proto_val)}")
        print(f"  Baud value     : {int(baud_val) if baud_val is not None else 'Unknown'}")
        print(f"  Likely device  : {infer_from_protocol(proto_val)}")
        print()

    return results


def sniff_messages(master, duration=10):
    print("\n=== LISTENING FOR MAVLINK MESSAGES ===\n")
    print(f"Listening for {duration} seconds...\n")

    seen_counts = defaultdict(int)

    start = time.time()
    while time.time() - start < duration:
        msg = master.recv_match(blocking=True, timeout=0.5)
        if not msg:
            continue

        mtype = msg.get_type()
        if mtype == "BAD_DATA":
            continue

        seen_counts[mtype] += 1

    if not seen_counts:
        print("No MAVLink messages received during the listen window.")
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
        count = seen_counts.get(mtype, 0)
        if count > 0:
            print(f"{mtype}: {count}")
            print(f"  Hint: {MESSAGE_HINTS.get(mtype, 'No hint available')}")
            print()

    print("Top message counts:")
    for mtype, count in sorted(seen_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {mtype}: {count}")

    return seen_counts


def summarize(serial_results, seen_counts):
    print("\n=== BEST-GUESS SUMMARY ===\n")

    for i, vals in serial_results.items():
        label = port_label(i).upper()
        guess = vals["guess"]
        proto = vals["protocol_text"]
        print(f"SERIAL{i} ({label}) -> {guess} [{proto}]")

    print()

    if seen_counts.get("RADIO_STATUS", 0) > 0:
        print("- Telemetry radio appears active because RADIO_STATUS messages were seen.")

    if seen_counts.get("DISTANCE_SENSOR", 0) > 0 or seen_counts.get("RANGEFINDER", 0) > 0:
        print("- Lidar / rangefinder appears active because distance messages were seen.")

    if seen_counts.get("GPS_RAW_INT", 0) > 0:
        print("- GPS appears active because GPS_RAW_INT messages were seen.")

    if (
        seen_counts.get("RADIO_STATUS", 0) == 0
        and seen_counts.get("DISTANCE_SENSOR", 0) == 0
        and seen_counts.get("RANGEFINDER", 0) == 0
        and seen_counts.get("GPS_RAW_INT", 0) == 0
    ):
        print("- No strong device-specific MAVLink messages were seen during the listen period.")

    print("\nNote:")
    print("This gives a best guess based on SERIALx configuration and live MAVLink traffic.")
    print("It does not guarantee the exact physical device model plugged into each port.")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Pixhawk serial ports and infer connected devices"
    )
    parser.add_argument("--port", default="/dev/ttyACM0", help="Pixhawk connection port")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--listen", type=int, default=10, help="Seconds to listen for MAVLink messages")
    args = parser.parse_args()

    print("=" * 60)
    print("PIXHAWK PORT INSPECTOR")
    print("=" * 60)
    print(f"Connecting to {args.port} @ {args.baud} ...")

    master = mavutil.mavlink_connection(args.port, baud=args.baud)

    print("Waiting for heartbeat...")
    master.wait_heartbeat(timeout=10)

    print("Heartbeat received.")
    print(f"System ID    : {master.target_system}")
    print(f"Component ID : {master.target_component}")

    serial_results = inspect_params(master, max_serial=6)
    seen_counts = sniff_messages(master, duration=args.listen)
    summarize(serial_results, seen_counts)


if __name__ == "__main__":
    main()
