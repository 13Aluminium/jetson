import time
from pymavlink import mavutil

master = mavutil.mavlink_connection('/dev/ttyACM0', baud=57600)
print("Waiting for heartbeat...")
msg = master.wait_heartbeat()
print(f"Connected! system={master.target_system} comp={master.target_component}")
print(f"Autopilot type: {msg.autopilot} (3=ArduPilot, 12=PX4)")
print(f"Vehicle type: {msg.type}")

# Force arm
print("\nForce arming...")
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0, 1, 21196, 0, 0, 0, 0, 0)

# Wait and check ACK
time.sleep(1)
ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
if ack:
    print(f"Arm ACK: result={ack.result} (0=success, 4=rejected)")
else:
    print("No ACK received")

# Check if actually armed
hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
if hb:
    armed = hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
    print(f"Actually armed: {bool(armed)}")

print("Done.")
