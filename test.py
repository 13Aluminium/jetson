import time
from pymavlink import mavutil

master = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)
print("Waiting for heartbeat...")
master.wait_heartbeat()
master.target_component = 1
print("Connected!")

time.sleep(1)

# Exact same as "arm throttle force" in MAVProxy
print("Arming...")
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0, 1, 21196, 0, 0, 0, 0, 0)

time.sleep(3)
print("Sending rc 3 1200...")

for i in range(100):
    master.mav.rc_channels_override_send(
        master.target_system, master.target_component,
        0, 0, 1200, 0, 0, 0, 0, 0)
    time.sleep(0.05)

time.sleep(3)

print("Stopping rc 3 1000...")
for i in range(20):
    master.mav.rc_channels_override_send(
        master.target_system, master.target_component,
        0, 0, 1000, 0, 0, 0, 0, 0)
    time.sleep(0.05)

time.sleep(1)

print("Disarming...")
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0, 0, 21196, 0, 0, 0, 0, 0)

print("Done.")
