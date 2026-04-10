import time
from pymavlink import mavutil

master = mavutil.mavlink_connection('/dev/ttyACM0', baud=57600)
print("Waiting for heartbeat...")
master.wait_heartbeat()
print(f"Connected! system={master.target_system} comp={master.target_component}")

# Fix component target to 1 (autopilot)
master.target_component = 1

# Check what's blocking arm
print("\nChecking prearm status...")
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_RUN_PREARM_CHECKS,
    0, 0, 0, 0, 0, 0, 0, 0)

# Read statustext messages for errors
start = time.time()
while time.time() - start < 5:
    msg = master.recv_match(type='STATUSTEXT', blocking=True, timeout=2)
    if msg:
        print(f"  >> {msg.text}")
    else:
        break

# Now force arm with correct component
print("\nForce arming (comp=1)...")
master.mav.command_long_send(
    master.target_system, 1,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0, 1, 21196, 0, 0, 0, 0, 0)

time.sleep(1)
ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
if ack:
    print(f"Arm ACK: result={ack.result}")

hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
if hb:
    armed = hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
    print(f"Armed: {bool(armed)}")

print("Done.")
