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

    # Send several setpoints before starting offboard
    print("Sending setpoints...")
    for i in range(20):
        await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.0))
        await asyncio.sleep(0.1)

    print("Starting offboard...")
    try:
        await drone.offboard.start()
        print("Offboard started!")
    except OffboardError as e:
        print(f"Offboard failed: {e}")
        return

    print("Arming...")
    await drone.action.arm()
    print("Armed!")

    # Now ramp up throttle
    print("Spinning motors...")
    for i in range(20):
        await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.2))
        await asyncio.sleep(0.1)

    # Hold for 5 seconds
    await asyncio.sleep(5)

    print("Stopping...")
    await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(1)
    await drone.action.kill()
    print("Done.")

asyncio.run(run())
