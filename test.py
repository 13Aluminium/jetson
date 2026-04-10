import time
from pymavlink import mavutil

DEVICE = "/dev/ttyACM0"
BAUD = 115200

def wait_cmd_ack(master, command_id, timeout=5):
    start = time.time()
    while time.time() - start < timeout:
        try:
            msg = master.recv_match(type="COMMAND_ACK", blocking=True, timeout=1)
            if msg and msg.command == command_id:
                return msg
        except:
            pass
    return None

master = mavutil.mavlink_connection(DEVICE, baud=BAUD)
print("Waiting for heartbeat...")
master.wait_heartbeat(timeout=30)
print(f"Connected! system={master.target_system}")

# Force arm
print("Force arming...")
master.mav.command_long_send(
    master.target_system, 1,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0, 1, 21196, 0, 0, 0, 0, 0)
ack = wait_cmd_ack(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)
print(f"ARM ACK: result={ack.result if ack else 'none'}")

time.sleep(1)

# Motor test — no armed check, ACK was success
for motor in [1, 2, 3, 4]:
    print(f"Testing motor {motor} at 15% for 2 seconds...")
    master.mav.command_long_send(
        master.target_system, 1,
        mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST,
        0,
        motor,  # motor number
        0,      # throttle type: percent
        15,     # throttle percent
        2,      # duration seconds
        1, 0, 0)
    ack = wait_cmd_ack(master, mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST)
    print(f"Motor {motor} ACK: result={ack.result if ack else 'none'}")
    time.sleep(3)

# Force disarm
print("Disarming...")
master.mav.command_long_send(
    master.target_system, 1,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0, 0, 21196, 0, 0, 0, 0, 0)
print("Done.")
