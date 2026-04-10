import time
from pymavlink import mavutil

# Use 115200 like MAVProxy
master = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)
print("Waiting for heartbeat...")
master.wait_heartbeat()
master.target_component = 1
print(f"Connected! system={master.target_system}")

# Set ARMING_CHECK to 0
master.mav.param_set_send(
    master.target_system, master.target_component,
    b'ARMING_CHECK', 0,
    mavutil.mavlink.MAV_PARAM_TYPE_INT32)
time.sleep(2)

# "arm throttle force" in MAVProxy sends this:
print("Arm throttle force...")
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,
    1,      # arm
    21196,  # force
    0, 0, 0, 0, 0)

time.sleep(2)

# Check armed
hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
armed = bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) if hb else False
print(f"Armed: {armed}")

# Check statustext for errors
start = time.time()
while time.time() - start < 3:
    msg = master.recv_match(type='STATUSTEXT', blocking=True, timeout=1)
    if msg:
        print(f"  >> {msg.text}")
    else:
        break

if armed:
    # "rc 3 2000" in MAVProxy
    print("RC 3 = 1200 (low throttle)...")
    for i in range(50):
        master.mav.rc_channels_override_send(
            master.target_system, master.target_component,
            0, 0, 1200, 0, 0, 0, 0, 0)
        time.sleep(0.1)

    time.sleep(5)

    print("Stopping...")
    master.mav.rc_channels_override_send(
        master.target_system, master.target_component,
        0, 0, 1000, 0, 0, 0, 0, 0)
    time.sleep(1)

    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, 21196, 0, 0, 0, 0, 0)

print("Done.")
