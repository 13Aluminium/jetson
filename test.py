import time
from pymavlink import mavutil

master = mavutil.mavlink_connection('/dev/ttyACM0', baud=57600)
print("Waiting for heartbeat...")
master.wait_heartbeat()
master.target_component = 1
print(f"Connected! system={master.target_system}")

# Disable arming checks
print("Disabling arming checks...")
master.mav.param_set_send(
    master.target_system, master.target_component,
    b'ARMING_CHECK', 0,
    mavutil.mavlink.MAV_PARAM_TYPE_INT32)
time.sleep(2)

# Verify param was set
master.mav.param_request_read_send(
    master.target_system, master.target_component,
    b'ARMING_CHECK', -1)
msg = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=5)
if msg:
    print(f"ARMING_CHECK = {msg.param_value}")

# Force arm
print("\nForce arming...")
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0, 1, 21196, 0, 0, 0, 0, 0)

time.sleep(1)
ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
if ack:
    print(f"Arm ACK: result={ack.result} (0=success)")

hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
if hb:
    armed = hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
    print(f"Armed: {bool(armed)}")

if armed:
    print("Spinning motors...")
    for i in range(50):
        master.mav.rc_channels_override_send(
            master.target_system, master.target_component,
            0, 0, 1200, 0, 0, 0, 0, 0)
        time.sleep(0.1)

    print("Stopping...")
    master.mav.rc_channels_override_send(
        master.target_system, master.target_component,
        0, 0, 1000, 0, 0, 0, 0, 0)
    time.sleep(1)

    print("Disarming...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, 21196, 0, 0, 0, 0, 0)

print("Done.")
