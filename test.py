import asyncio
from mavsdk import System
from mavsdk.offboard import Attitude, OffboardError

async def run():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyACM0:57600")

    # Wait for connection
    print("Waiting for drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected!")
            break

    # Wait for position estimate
    print("Waiting for position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("Position OK")
            break

    # Set initial setpoint before starting offboard
    await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.15))

    # Arm
    print("Arming...")
    await drone.action.arm()
    print("Armed!")

    # Start offboard
    print("Starting offboard...")
    await drone.offboard.start()
    print("Motors spinning!")

    # Run for 5 seconds
    await asyncio.sleep(5)

    # Stop
    print("Stopping...")
    await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(0.5)
    await drone.offboard.stop()
    await drone.action.disarm()
    print("Done.")

asyncio.run(run())
