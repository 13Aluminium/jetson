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

# Ask ArduPilot to skip pre-arm checks for bench testing only
# ARMING_SKIPCHK = 1 means skip all checks
master.mav.param_set_send(
    master.target_system,
    master.target_component or 1,
    b"ARMING_SKIPCHK",
    float(1),
    mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
)
time.sleep(1)

print("Force arming...")
master.mav.command_long_send(
    master.target_system,
    master.target_component or 1,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,
    1,        # arm
    21196,    # force
    0, 0, 0, 0, 0
)

ack = wait_cmd_ack(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
print("ARM ACK:", ack)

# Check heartbeat armed flag
hb = master.recv_match(type="HEARTBEAT", blocking=True, timeout=5)
armed = bool(hb and (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED))
print("Armed:", armed)

if armed:
    print("Testing motor 1 at 10% for 2 seconds...")

    # MAV_CMD_DO_MOTOR_TEST params:
    # p1 = motor number (1-based)
    # p2 = throttle type (0 = percent)
    # p3 = throttle value
    # p4 = timeout seconds
    # p5 = motor count / sequence control (use 1 here)
    # p6,p7 unused
    master.mav.command_long_send(
        master.target_system,
        master.target_component or 1,
        mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST,
        0,
        1,    # motor 1
        0,    # throttle type: percent
        10,   # 10%
        2,    # 2 seconds
        1,    # test this one motor
        0, 0
    )

    motor_ack = wait_cmd_ack(master, mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST)
    print("MOTOR TEST ACK:", motor_ack)

    time.sleep(3)

print("Disarming...")
master.mav.command_long_send(
    master.target_system,
    master.target_component or 1,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,
    0,        # disarm
    21196,    # force
    0, 0, 0, 0, 0
)

ack = wait_cmd_ack(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
print("DISARM ACK:", ack)
print("Done.")
