# ROS 2 Humble: X-target guidance

These scripts publish the existing X-target guidance as a JSON string on
`/x_target/guidance`. The message contains `target_detected`,
`horizontal_offset_px`, `vertical_offset_px`, `confidence`, `centered`,
and `guidance`. Offsets are relative to the camera-frame center: positive
horizontal means right; positive vertical means below. Offsets are null when
no target is detected. The app publishes at 10 Hz and changes to
`target_detected: false` if detections stop.

## Install and source ROS 2 Humble

On Ubuntu with ROS 2 Humble installed, open each terminal and run:

```bash
source /opt/ros/humble/setup.bash
```

The ROS package dependency used by these scripts is:

```bash
sudo apt install ros-humble-rclpy ros-humble-std-msgs
```

The Jetson's ROS 2 installation and camera/model dependencies must be checked
on the device separately. These instructions do not assume ROS 2 is installed
there.

## Test without Jetson hardware

In terminal 1, start the changing sample publisher:

```bash
source /opt/ros/humble/setup.bash
python3 simulated_guidance_publisher.py
```

In terminal 2, run the simple subscriber:

```bash
source /opt/ros/humble/setup.bash
python3 x_guidance_listener.py
```

The sample publisher cycles through an off-center detection, a centered
detection, and no target. No camera, YOLO model, Pixhawk, or motors are used.
You can also inspect the topic with:

```bash
ros2 topic echo /x_target/guidance
```

## Run the existing vision app with ROS publishing

Keep `x_detect_guide.py`, `x_detect_guide_ros2.py`, and the YOLO weights in
the same directory. In a terminal with ROS 2 sourced, run:

```bash
python3 x_detect_guide_ros2.py --weights best_22.pt
```

This keeps the existing live OpenCV window and terminal guidance and publishes
to `/x_target/guidance`. Use `--headless` for terminal-only camera operation,
or `--refine` to use the existing green-segmentation center estimate. Example:

```bash
python3 x_detect_guide_ros2.py --weights best_22.pt --headless --refine --deadzone 80
```

The wrapper only publishes guidance; it does not send movement or motor
commands. The ROS-enabled camera run requires the existing OpenCV, camera,
Ultralytics, and model setup in addition to ROS 2.
