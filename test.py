import time
from pymavlink import mavutil

master = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)
print("Waiting for heartbeat...")
master.wait_heartbeat()
print("Connected!")
print("Done.")
