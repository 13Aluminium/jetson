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

    # Disable arming checks via params
    await drone.param.set_param_int("COM_ARM_WO_GPS", 1)
    await drone.param.set_param_int("COM_ARM_EKF_CHECK", 0)
    print("Arming checks disabled")

    await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.15))

    print("Arming...")
    await drone.action.arm()
    print("Armed!")

    print("Starting offboard...")
    await drone.offboard.start()
    print("Motors spinning!")

    await asyncio.sleep(5)

    print("Stopping...")
    await drone.offboard.set_attitude(Attitude(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(0.5)
    await drone.offboard.stop()
    await drone.action.disarm()
    print("Done.")

asyncio.run(run())
