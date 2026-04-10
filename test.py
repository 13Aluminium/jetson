import time
from pymavlink import mavutil

master = mavutil.mavlink_connection('/dev/ttyACM0', baud=57600)
print("Waiting for heartbeat...")
master.wait_heartbeat()
print(f"Connected! (system {master.target_system}, comp {master.target_component})")

# Force arm — 21196 bypasses all checks
print("Force arming...")
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0, 1, 21196, 0, 0, 0, 0, 0)
time.sleep(2)
print("Arm command sent!")

# Send low throttle via RC override
print("Spinning motors...")
for i in range(100):
    master.mav.rc_channels_override_send(
        master.target_system, master.target_component,
        0, 0, 1200, 0, 0, 0, 0, 0)
    time.sleep(0.1)

# Stop
print("Stopping...")
master.mav.rc_channels_override_send(
    master.target_system, master.target_component,
    0, 0, 1000, 0, 0, 0, 0, 0)
time.sleep(1)

# Force disarm
print("Force disarming...")
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0, 0, 21196, 0, 0, 0, 0, 0)

print("Done.")
