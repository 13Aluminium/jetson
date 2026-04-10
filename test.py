import time
from pymavlink import mavutil

DEVICE = "/dev/ttyACM0"
BAUD = 115200

def wait_cmd_ack(master, command_id, timeout=5):
    start = time.time()
    while time.time() - start < timeout:
        msg = master.recv_match(type="COMMAND_ACK", blocking=True, timeout=1)
        if msg and msg.command == command_id:
            return msg
    return None

master = mavutil.mavlink_connection(DEVICE, baud=BAUD)
print("Waiting for heartbeat...")
master.wait_heartbeat(timeout=30)
print(f"Connected! system={master.target_system}, component={master.target_component}")

print("Force arming...")
master.mav.command_long_send(
    master.target_system,
    1,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,
    1, 21196, 0, 0, 0, 0, 0
)

ack = wait_cmd_ack(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
print("ARM ACK:", ack)

hb = master.recv_match(type="HEARTBEAT", blocking=True, timeout=5)
armed = bool(hb and (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED))
print("Armed:", armed)

if armed:
    for motor in [1, 2, 3, 4]:
        print(f"Testing motor {motor} at 15% for 2 seconds...")
        master.mav.command_long_send(
            master.target_system,
            1,
            mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST,
            0,
            motor,  # motor number
            0,      # throttle type: percent
            15,     # throttle percent
            2,      # duration in seconds
            1,
            0, 0
        )
        ack = wait_cmd_ack(master, mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST)
        print(f"Motor {motor} ACK:", ack)
        time.sleep(3)

print("Disarming...")
master.mav.command_long_send(
    master.target_system,
    1,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,
    0, 21196, 0, 0, 0, 0, 0
)

ack = wait_cmd_ack(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
print("DISARM ACK:", ack)
print("Done.")
