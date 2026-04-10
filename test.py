import asyncio
from mavsdk import System

async def run():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyACM0:57600")

    print("Waiting for drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected!")
            break

    # Give it a moment to sync
    await asyncio.sleep(3)

    print("Arming...")
    try:
        await drone.action.arm()
        print("Armed!")
    except Exception as e:
        print(f"Arm failed: {e}")

    await asyncio.sleep(5)

    print("Disarming...")
    await drone.action.disarm()
    print("Done.")

asyncio.run(run())
