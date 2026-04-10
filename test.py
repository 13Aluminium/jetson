import time
from pymavlink import mavutil

master = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)
print("Waiting for heartbeat...")
master.wait_heartbeat()
master.target_component = 1
print(f"Connected! system={master.target_system}")

# Disable arming checks
master.mav.param_set_send(
    master.target_system, master.target_component,
    b'ARMING_CHECK', 0,
    mavutil.mavlink.MAV_PARAM_TYPE_INT32)
time.sleep(2)

# Force arm
print("Arm throttle force...")
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0, 1, 21196, 0, 0, 0, 0, 0)

# Wait and drain messages safely
time.sleep(3)
try:
    while True:
        msg = master.recv_match(blocking=False)
        if msg is None:
            break
except:
    pass

# Check armed via heartbeat
try:
    hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
    if hb:
        armed = bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        print(f"Armed: {armed}")
    else:
        print("No heartbeat — assuming armed, sending throttle anyway")
        armed = True
except:
    print("Recv error — sending throttle anyway")
    armed = True

if armed:
    print("RC 3 = 1200...")
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
