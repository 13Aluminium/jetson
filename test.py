import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, ActuatorControl

async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected!")
            break

    print("Waiting for global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_armable:
            print("Drone is armable")
            break

    print("Arming...")
    await drone.action.arm()

    # Neutral actuator values (0 = no movement)
    neutral = [0.0] * 8

    try:
        print("Starting offboard mode...")
        await drone.offboard.set_actuator_control(ActuatorControl([neutral, neutral]))
        await drone.offboard.start()
    except OffboardError as e:
        print(f"Offboard start failed: {e}")
        await drone.action.disarm()
        return

    print("Spinning motors slowly...")
    # Small throttle value (0.1 = very low power)
    motor_test = [0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]

    await drone.offboard.set_actuator_control(
        ActuatorControl([motor_test, neutral])
    )

    await asyncio.sleep(5)  # run motors for 5 seconds

    print("Stopping motors...")
    await drone.offboard.set_actuator_control(
        ActuatorControl([neutral, neutral])
    )

    await drone.offboard.stop()

    print("Disarming...")
    await drone.action.disarm()

if __name__ == "__main__":
    asyncio.run(run())