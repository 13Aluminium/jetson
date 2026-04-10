import time
from pymavlink import mavutil

# Connect to Pixhawk
master = mavutil.mavlink_connection('/dev/ttyACM0', baud=57600)
print("Waiting for heartbeat...")
master.wait_heartbeat()
print(f"Connected! (system {master.target_system}, comp {master.target_component})")

# Force arm (bypass all checks)
print("Arming...")
master.arducopter_arm()
# Or if that doesn't work:
# master.mav.command_long_send(
#     master.target_system, master.target_component,
#     mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
#     0, 1, 21196, 0, 0, 0, 0, 0)  # 21196 = force arm

master.motors_armed_wait()
print("Armed!")

# Send throttle via RC override (channel 3 = throttle)
# 1000 = min, 1500 = mid, 2000 = max
print("Spinning motors at low throttle...")
for i in range(50):
    master.mav.rc_channels_override_send(
        master.target_system, master.target_component,
        0, 0, 1200, 0, 0, 0, 0, 0)  # ch3=1200 (low throttle)
    time.sleep(0.1)

print("Motors running for 5 seconds...")
time.sleep(5)

# Stop
print("Stopping...")
master.mav.rc_channels_override_send(
    master.target_system, master.target_component,
    0, 0, 1000, 0, 0, 0, 0, 0)  # ch3=1000 (min)
time.sleep(1)

# Disarm
print("Disarming...")
master.arducopter_disarm()
# Force disarm if needed:
# master.mav.command_long_send(
#     master.target_system, master.target_component,
#     mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
#     0, 0, 21196, 0, 0, 0, 0, 0)

print("Done.")
