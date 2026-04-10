import asyncio
from mavsdk import System
from mavsdk.offboard import Attitude, OffboardError

async def run():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyACM0:57600")

    print("Waiting for drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected!")
            break

    await asyncio.sleep(3)

    # Set initial setpoint before starting offboard
    print("Setting initial setpoint...")
    await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.2))

    print("Arming...")
    await drone.action.arm()
    print("Armed!")

    print("Starting offboard...")
    try:
        await drone.offboard.start()
        print("Offboard started — motors should spin!")
    except OffboardError as e:
        print(f"Offboard failed: {e}")
        await drone.action.kill()
        return

    # Spin for 5 seconds
    await asyncio.sleep(5)

    # Throttle down then kill
    print("Stopping...")
    await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(1)
    try:
        await drone.offboard.stop()
    except:
        pass
    await drone.action.kill()
    print("Done.")

asyncio.run(run())
